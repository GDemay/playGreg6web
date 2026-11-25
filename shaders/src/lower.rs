use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use naga::back::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, AtomicFunction, Block, Expression, Function, Handle, ImageClass,
    ImageDimension, ImageQuery, MemoryDecorations, ResourceBinding, SampleLevel, ScalarKind,
    ShaderStage, Statement, StorageAccess, StorageFormat, Type, TypeInner,
};

use crate::compat::WEBGPU_MAX_BIND_GROUPS;

/// Preferred WebGPU bind group for lowered push constants / immediate data.
pub const PUSH_CONSTANT_TARGET_GROUP: u32 = WEBGPU_MAX_BIND_GROUPS - 1;

/// WebGPU baseline default `maxBindingsPerBindGroup`.
pub const WEBGPU_MAX_BINDINGS_PER_GROUP: usize = 1000;

/// Maximum type nesting depth permitted during lowering validation (prevents cycles / stack overflow).
const MAX_TYPE_DEPTH: usize = 64;

/// Maximum statement block nesting depth permitted during lowering validation.
const MAX_BLOCK_DEPTH: usize = 128;

/// Maximum pointer-dereference chain length when tracing base globals for stores/atomics.
const MAX_POINTER_TRACE_DEPTH: usize = 64;

/// Record of an `AddressSpace::Immediate` global variable lowered into a WebGPU uniform buffer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PushConstantRemap {
    pub variable_name: Option<String>,
    pub new_group: u32,
    pub new_binding: u32,
}

/// Record of an out-of-range (`group >= 4`) resource binding compressed into groups `0..=3`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BindGroupRemap {
    pub variable_name: Option<String>,
    pub old_group: u32,
    pub old_binding: u32,
    pub new_group: u32,
    pub new_binding: u32,
}

/// Summary of all Naga IR WebGPU lowering rewrites performed on a module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoweringReport {
    pub push_constant_remaps: Vec<PushConstantRemap>,
    pub bind_group_remaps: Vec<BindGroupRemap>,
}

impl LoweringReport {
    /// Total number of global variable rewrites performed across all lowering passes.
    pub fn total_remaps(&self) -> usize {
        self.push_constant_remaps.len() + self.bind_group_remaps.len()
    }

    /// Returns `true` if no lowering rewrites were needed.
    pub fn is_empty(&self) -> bool {
        self.push_constant_remaps.is_empty() && self.bind_group_remaps.is_empty()
    }

    /// Return a copy with all remap records sorted in canonical deterministic order.
    pub fn canonicalized(&self) -> Self {
        let mut out = self.clone();
        out.push_constant_remaps.sort_by(|a, b| {
            (
                a.new_group,
                a.new_binding,
                a.variable_name.is_none(),
                a.variable_name.as_deref().unwrap_or(""),
            )
                .cmp(&(
                    b.new_group,
                    b.new_binding,
                    b.variable_name.is_none(),
                    b.variable_name.as_deref().unwrap_or(""),
                ))
        });
        out.bind_group_remaps.sort_by(|a, b| {
            (
                a.old_group,
                a.old_binding,
                a.new_group,
                a.new_binding,
                a.variable_name.is_none(),
                a.variable_name.as_deref().unwrap_or(""),
            )
                .cmp(&(
                    b.old_group,
                    b.old_binding,
                    b.new_group,
                    b.new_binding,
                    b.variable_name.is_none(),
                    b.variable_name.as_deref().unwrap_or(""),
                ))
        });
        out
    }
}

/// Structured diagnostic returned when a Naga IR module cannot be safely lowered to baseline WebGPU WGSL.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoweringError {
    /// A handle referenced an out-of-bounds entry in a Naga arena.
    InvalidHandle {
        arena: &'static str,
        index: usize,
        len: usize,
        context: String,
    },
    /// A `Statement::Emit` expression range fell outside the function's expression arena.
    InvalidExpressionRange {
        function: String,
        start: usize,
        end: usize,
        expr_count: usize,
    },
    /// Type definitions form a cycle or exceed the maximum supported nesting depth.
    CyclicOrDeepType {
        type_index: usize,
        max_depth: usize,
    },
    /// Statement block nesting exceeded the maximum supported depth.
    BlockDepthExceeded {
        function: String,
        max_depth: usize,
    },
    /// Struct or array type is zero-sized, has zero stride, invalid member layout, or overflows `u32`.
    ZeroSizedOrInvalidType {
        type_name: String,
        detail: String,
    },
    /// Resource global is missing `@group/@binding`, or a non-resource global carries a binding.
    InvalidResourceBinding {
        variable: String,
        detail: String,
    },
    /// Two distinct resource globals collide on the same `(group, binding)` slot.
    DuplicateResourceBinding {
        group: u32,
        binding: u32,
        first_variable: String,
        second_variable: String,
    },
    /// Target bind group has no free `u32` binding slot available for remapping.
    BindingSlotExhausted {
        group: u32,
        variable: String,
        detail: String,
    },
    /// Push constant (`AddressSpace::Immediate`) uses a type or initializer incompatible with `var<uniform>`.
    UnsupportedPushConstantType {
        variable: String,
        detail: String,
    },
    /// Storage buffer declaration or store operation violates baseline WebGPU constraints.
    InvalidStorageOp {
        context: String,
        detail: String,
    },
    /// Texture declaration or texture operation violates baseline WebGPU constraints.
    InvalidTextureOp {
        context: String,
        detail: String,
    },
    /// Atomic declaration or atomic operation violates baseline WebGPU constraints.
    InvalidAtomicOp {
        context: String,
        detail: String,
    },
    /// Derivative operation (`dpdx`/`dpdy`/`fwidth`) is invalid for the reachable shader stage.
    InvalidDerivativeOp {
        context: String,
        detail: String,
    },
    /// Naga validation rejected the lowered module under baseline WebGPU capabilities.
    ValidationFailed(String),
    /// Naga WGSL backend failed to emit WGSL source for the validated module.
    WgslEmitFailed(String),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle {
                arena,
                index,
                len,
                context,
            } => write!(
                f,
                "invalid handle in {arena} arena (index {index}, len {len}) while inspecting {context}"
            ),
            Self::InvalidExpressionRange {
                function,
                start,
                end,
                expr_count,
            } => write!(
                f,
                "invalid Statement::Emit expression range [{start}..={end}] in '{function}' (expression arena has {expr_count} entries)"
            ),
            Self::CyclicOrDeepType {
                type_index,
                max_depth,
            } => write!(
                f,
                "type #{type_index} is cyclic or exceeds maximum nesting depth ({max_depth})"
            ),
            Self::BlockDepthExceeded {
                function,
                max_depth,
            } => write!(
                f,
                "statement nesting in '{function}' exceeds maximum depth ({max_depth})"
            ),
            Self::ZeroSizedOrInvalidType { type_name, detail } => {
                write!(f, "invalid/zero-sized type '{type_name}': {detail}")
            }
            Self::InvalidResourceBinding { variable, detail } => {
                write!(f, "invalid resource binding on '{variable}': {detail}")
            }
            Self::DuplicateResourceBinding {
                group,
                binding,
                first_variable,
                second_variable,
            } => write!(
                f,
                "duplicate @group({group}) @binding({binding}) collision between '{first_variable}' and '{second_variable}'"
            ),
            Self::BindingSlotExhausted {
                group,
                variable,
                detail,
            } => write!(
                f,
                "binding slot allocation exhausted in group {group} for '{variable}': {detail}"
            ),
            Self::UnsupportedPushConstantType { variable, detail } => write!(
                f,
                "push-constant variable '{variable}' cannot be lowered to uniform buffer: {detail}"
            ),
            Self::InvalidStorageOp { context, detail } => {
                write!(f, "invalid storage operation in {context}: {detail}")
            }
            Self::InvalidTextureOp { context, detail } => {
                write!(f, "invalid texture operation in {context}: {detail}")
            }
            Self::InvalidAtomicOp { context, detail } => {
                write!(f, "invalid atomic operation in {context}: {detail}")
            }
            Self::InvalidDerivativeOp { context, detail } => {
                write!(f, "invalid derivative operation in {context}: {detail}")
            }
            Self::ValidationFailed(detail) => {
                write!(f, "baseline WebGPU validation failed after lowering: {detail}")
            }
            Self::WgslEmitFailed(detail) => {
                write!(f, "WGSL emission failed after lowering: {detail}")
            }
        }
    }
}

impl std::error::Error for LoweringError {}

/// Run all WebGPU baseline Naga IR lowering passes in-place on `module`:
///
/// 1. **Pass 1 — Push Constants (`AddressSpace::Immediate`) -> Uniform Buffer:**
///    Rewrites immediate/push-constant globals into `AddressSpace::Uniform` bound in
///    group `3` (or the lowest free group `< 4` if group `3` is full).
/// 2. **Pass 2 — Bind-Group Compression (`group >= 4` -> `0..=3`):**
///    Deterministically remaps every binding with `group >= 4` into the least-populated
///    group in `0..=3` using `max_existing_binding + 1`.
///
/// Intended to run immediately after SPIR-V front-end parsing and BEFORE `naga::valid::Validator`
/// and WGSL emission. Never panics on empty modules, malformed handles, or saturated groups.
pub fn lower_module_for_webgpu(module: &mut naga::Module) -> LoweringReport {
    let push_constant_remaps = lower_push_constants(module);
    let bind_group_remaps = lower_bind_groups(module);
    LoweringReport {
        push_constant_remaps,
        bind_group_remaps,
    }
}

/// Hardened, transactional version of [`lower_module_for_webgpu`].
///
/// Audits types, bindings, ranges, storage, texture, atomic, and derivative operations before
/// rewriting the module. If any check or lowering pass fails, `module` is restored to its
/// original state and a structured [`LoweringError`] is returned.
pub fn try_lower_module_for_webgpu(
    module: &mut naga::Module,
) -> Result<LoweringReport, LoweringError> {
    validate_module_for_lowering(module)?;

    let backup = module.clone();
    let result = (|| {
        let push_constant_remaps = try_lower_push_constants_inner(module)?;
        let bind_group_remaps = try_lower_bind_groups_inner(module)?;
        Ok(LoweringReport {
            push_constant_remaps,
            bind_group_remaps,
        })
    })();

    if result.is_err() {
        *module = backup;
    }
    result
}

/// Transactionally lower `module`, validate under strict baseline WebGPU capabilities
/// (`Capabilities::empty()`), and emit deterministic WGSL source.
///
/// On failure, `module` is rolled back to its input state and a structured [`LoweringError`]
/// is returned.
pub fn lower_and_emit_wgsl(
    module: &mut naga::Module,
) -> Result<(String, LoweringReport), LoweringError> {
    let backup = module.clone();
    let report = match try_lower_module_for_webgpu(module) {
        Ok(report) => report,
        Err(err) => {
            *module = backup;
            return Err(err);
        }
    };

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = match validator.validate(module) {
        Ok(info) => info,
        Err(err) => {
            *module = backup;
            return Err(LoweringError::ValidationFailed(err.emit_to_string("")));
        }
    };

    match wgsl::write_string(module, &info, wgsl::WriterFlags::empty()) {
        Ok(wgsl_source) => Ok((wgsl_source, report)),
        Err(err) => {
            *module = backup;
            Err(LoweringError::WgslEmitFailed(err.to_string()))
        }
    }
}

/// Pass 1: Lower any `AddressSpace::Immediate` global variables (D3D12 root constants / Vulkan
/// push constants) to `AddressSpace::Uniform` bindings in WebGPU-valid bind groups `0..=3`.
pub fn lower_push_constants(module: &mut naga::Module) -> Vec<PushConstantRemap> {
    let mut used = collect_used_bindings_in_webgpu_groups(module);
    let immediate_handles = collect_sorted_immediate_handles(module);

    if immediate_handles.is_empty() {
        return Vec::new();
    }

    let mut remaps = Vec::with_capacity(immediate_handles.len());
    for handle in immediate_handles {
        let group = select_push_constant_group(&used);
        let Some(binding) = allocate_next_binding_checked(&mut used[group as usize]) else {
            continue;
        };

        if let Some(var) = lookup_global_mut(module, handle) {
            var.space = AddressSpace::Uniform;
            var.binding = Some(ResourceBinding { group, binding });
            var.memory_decorations = MemoryDecorations::empty();

            remaps.push(PushConstantRemap {
                variable_name: var.name.clone(),
                new_group: group,
                new_binding: binding,
            });
        }
    }

    // Rewrite standalone `Pointer`/`ValuePointer` types with `AddressSpace::Immediate`
    // into `AddressSpace::Uniform`, remapping references if an identical `Uniform` pointer
    // type already exists in `module.types`.
    rewrite_immediate_pointer_types(module);

    remaps
}

/// Checked, transactional version of [`lower_push_constants`].
pub fn try_lower_push_constants(
    module: &mut naga::Module,
) -> Result<Vec<PushConstantRemap>, LoweringError> {
    let backup = module.clone();
    let result = try_lower_push_constants_inner(module);
    if result.is_err() {
        *module = backup;
    }
    result
}

fn try_lower_push_constants_inner(
    module: &mut naga::Module,
) -> Result<Vec<PushConstantRemap>, LoweringError> {
    let mut used = collect_used_bindings_in_webgpu_groups(module);
    let immediate_handles = collect_sorted_immediate_handles(module);

    if immediate_handles.is_empty() {
        return Ok(Vec::new());
    }

    // Validate all immediate globals before mutating anything.
    for &handle in &immediate_handles {
        let Some(var) = lookup_global(module, handle) else {
            return Err(LoweringError::InvalidHandle {
                arena: "global_variables",
                index: handle.index(),
                len: module.global_variables.len(),
                context: "push-constant lowering".to_string(),
            });
        };
        validate_push_constant_global(module, handle, var)?;
    }

    let mut remaps = Vec::with_capacity(immediate_handles.len());
    for handle in immediate_handles {
        let var_label = lookup_global(module, handle)
            .map(|v| global_label(v, handle))
            .unwrap_or_else(|| format!("#{}", handle.index()));
        let group = select_push_constant_group(&used);
        let Some(binding) = allocate_next_binding_checked(&mut used[group as usize]) else {
            return Err(LoweringError::BindingSlotExhausted {
                group,
                variable: var_label,
                detail: "no free u32 binding slot remains in target bind group".to_string(),
            });
        };

        let Some(var) = lookup_global_mut(module, handle) else {
            return Err(LoweringError::InvalidHandle {
                arena: "global_variables",
                index: handle.index(),
                len: module.global_variables.len(),
                context: "push-constant rewrite".to_string(),
            });
        };
        var.space = AddressSpace::Uniform;
        var.binding = Some(ResourceBinding { group, binding });
        var.memory_decorations = MemoryDecorations::empty();

        remaps.push(PushConstantRemap {
            variable_name: var.name.clone(),
            new_group: group,
            new_binding: binding,
        });
    }

    rewrite_immediate_pointer_types(module);

    Ok(remaps)
}

/// Pass 2: Deterministically compress any global variable binding with `group >= 4` into free
/// binding slots inside WebGPU baseline groups `0..=3`.
pub fn lower_bind_groups(module: &mut naga::Module) -> Vec<BindGroupRemap> {
    let mut used = collect_used_bindings_in_webgpu_groups(module);
    let overflow_bindings = collect_sorted_overflow_bindings(module);

    if overflow_bindings.is_empty() {
        return Vec::new();
    }

    let mut remaps = Vec::with_capacity(overflow_bindings.len());
    for (old_group, old_binding, handle) in overflow_bindings {
        // Choose the least-populated group in 0..=3, tie-breaking by lowest group index.
        let new_group = select_least_populated_group(&used);
        let Some(new_binding) = allocate_next_binding_checked(&mut used[new_group as usize]) else {
            continue;
        };

        if let Some(var) = lookup_global_mut(module, handle) {
            var.binding = Some(ResourceBinding {
                group: new_group,
                binding: new_binding,
            });

            remaps.push(BindGroupRemap {
                variable_name: var.name.clone(),
                old_group,
                old_binding,
                new_group,
                new_binding,
            });
        }
    }

    remaps
}

/// Checked, transactional version of [`lower_bind_groups`].
pub fn try_lower_bind_groups(
    module: &mut naga::Module,
) -> Result<Vec<BindGroupRemap>, LoweringError> {
    let backup = module.clone();
    let result = try_lower_bind_groups_inner(module);
    if result.is_err() {
        *module = backup;
    }
    result
}

fn try_lower_bind_groups_inner(
    module: &mut naga::Module,
) -> Result<Vec<BindGroupRemap>, LoweringError> {
    validate_binding_collisions(module)?;

    let mut used = collect_used_bindings_in_webgpu_groups(module);
    let overflow_bindings = collect_sorted_overflow_bindings(module);

    if overflow_bindings.is_empty() {
        return Ok(Vec::new());
    }

    let mut remaps = Vec::with_capacity(overflow_bindings.len());
    for (old_group, old_binding, handle) in overflow_bindings {
        let var_label = lookup_global(module, handle)
            .map(|v| global_label(v, handle))
            .unwrap_or_else(|| format!("#{}", handle.index()));
        let new_group = select_least_populated_group(&used);
        let Some(new_binding) = allocate_next_binding_checked(&mut used[new_group as usize]) else {
            return Err(LoweringError::BindingSlotExhausted {
                group: new_group,
                variable: var_label,
                detail: "no free u32 binding slot remains in target bind group".to_string(),
            });
        };

        let Some(var) = lookup_global_mut(module, handle) else {
            return Err(LoweringError::InvalidHandle {
                arena: "global_variables",
                index: handle.index(),
                len: module.global_variables.len(),
                context: "bind-group compression".to_string(),
            });
        };
        var.binding = Some(ResourceBinding {
            group: new_group,
            binding: new_binding,
        });

        remaps.push(BindGroupRemap {
            variable_name: var.name.clone(),
            old_group,
            old_binding,
            new_group,
            new_binding,
        });
    }

    Ok(remaps)
}

// ---------------------------------------------------------------------------
// Pre-lowering structural, type, binding & op validation (zero-panic guards)
// ---------------------------------------------------------------------------

/// Audit a Naga IR module for structural, type, binding, storage, texture, atomic, and
/// derivative issues before lowering.
///
/// Guarantees zero `unwrap()`/`expect()` panics on malformed or edge-case IR.
pub fn validate_module_for_lowering(module: &naga::Module) -> Result<(), LoweringError> {
    // 1. Validate every type definition (bounds, cycles, zero-sized structs, zero-stride arrays).
    let mut active_types = BTreeSet::new();
    let mut validated_types = BTreeSet::new();
    for (handle, _) in module.types.iter() {
        validate_type_handle(
            module,
            handle,
            0,
            &mut active_types,
            &mut validated_types,
            "module.types",
        )?;
    }

    // 2. Validate constants and overrides type/init handles.
    for (handle, constant) in module.constants.iter() {
        let ctx = format!(
            "constant '{}'",
            constant.name.as_deref().unwrap_or("<unnamed>")
        );
        ensure_type_handle(module, constant.ty, &ctx)?;
        ensure_global_expr_handle(module, constant.init, &ctx)?;
        let _ = handle;
    }
    for (_, ov) in module.overrides.iter() {
        let ctx = format!("override '{}'", ov.name.as_deref().unwrap_or("<unnamed>"));
        ensure_type_handle(module, ov.ty, &ctx)?;
        if let Some(init) = ov.init {
            ensure_global_expr_handle(module, init, &ctx)?;
        }
    }

    // 3. Validate global variable declarations, bindings, address spaces, textures & atomics.
    for (handle, var) in module.global_variables.iter() {
        validate_global_variable(module, handle, var)?;
    }

    // 4. Validate static binding collisions (per entry point, or module-wide when no entry points).
    validate_binding_collisions(module)?;

    // 5. Build deterministic call graph and reachable shader stage sets (strictly BTreeMap/BTreeSet).
    let stage_reachability = compute_function_stage_reachability(module)?;

    // 6. Validate helper functions and entry points (expression handles, Emit ranges,
    //    storage/texture/atomic/derivative ops).
    for (fn_handle, func) in module.functions.iter() {
        let fn_ref = FnRef::Function(fn_handle);
        let stages = stage_reachability
            .get(&fn_ref)
            .cloned()
            .unwrap_or_default();
        let fn_name = func
            .name
            .clone()
            .unwrap_or_else(|| format!("function#{}", fn_handle.index()));
        validate_function(module, func, &fn_name, &stages)?;
    }
    for (ep_idx, ep) in module.entry_points.iter().enumerate() {
        let mut stages = BTreeSet::new();
        stages.insert(ep.stage);
        let fn_name = format!("entry_point '{}'", ep.name);
        validate_function(module, &ep.function, &fn_name, &stages)?;
        let _ = ep_idx;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FnRef {
    Function(Handle<Function>),
    Entry(usize),
}

fn lookup_type(module: &naga::Module, handle: Handle<Type>) -> Option<&Type> {
    (handle.index() < module.types.len()).then(|| &module.types[handle])
}

fn lookup_global(
    module: &naga::Module,
    handle: Handle<naga::GlobalVariable>,
) -> Option<&naga::GlobalVariable> {
    (handle.index() < module.global_variables.len()).then(|| &module.global_variables[handle])
}

fn lookup_global_mut(
    module: &mut naga::Module,
    handle: Handle<naga::GlobalVariable>,
) -> Option<&mut naga::GlobalVariable> {
    (handle.index() < module.global_variables.len()).then(|| &mut module.global_variables[handle])
}

fn lookup_expr(func: &Function, handle: Handle<Expression>) -> Option<&Expression> {
    (handle.index() < func.expressions.len()).then(|| &func.expressions[handle])
}

fn ensure_type_handle<'a>(
    module: &'a naga::Module,
    handle: Handle<Type>,
    context: &str,
) -> Result<&'a Type, LoweringError> {
    lookup_type(module, handle).ok_or_else(|| LoweringError::InvalidHandle {
        arena: "types",
        index: handle.index(),
        len: module.types.len(),
        context: context.to_string(),
    })
}

fn ensure_global_handle<'a>(
    module: &'a naga::Module,
    handle: Handle<naga::GlobalVariable>,
    context: &str,
) -> Result<&'a naga::GlobalVariable, LoweringError> {
    lookup_global(module, handle).ok_or_else(|| LoweringError::InvalidHandle {
        arena: "global_variables",
        index: handle.index(),
        len: module.global_variables.len(),
        context: context.to_string(),
    })
}

fn ensure_global_expr_handle(
    module: &naga::Module,
    handle: Handle<Expression>,
    context: &str,
) -> Result<(), LoweringError> {
    if handle.index() < module.global_expressions.len() {
        Ok(())
    } else {
        Err(LoweringError::InvalidHandle {
            arena: "global_expressions",
            index: handle.index(),
            len: module.global_expressions.len(),
            context: context.to_string(),
        })
    }
}

fn ensure_func_expr_handle(
    func: &Function,
    handle: Handle<Expression>,
    context: &str,
) -> Result<(), LoweringError> {
    if handle.index() < func.expressions.len() {
        Ok(())
    } else {
        Err(LoweringError::InvalidHandle {
            arena: "expressions",
            index: handle.index(),
            len: func.expressions.len(),
            context: context.to_string(),
        })
    }
}

fn ensure_function_handle(
    module: &naga::Module,
    handle: Handle<Function>,
    context: &str,
) -> Result<(), LoweringError> {
    if handle.index() < module.functions.len() {
        Ok(())
    } else {
        Err(LoweringError::InvalidHandle {
            arena: "functions",
            index: handle.index(),
            len: module.functions.len(),
            context: context.to_string(),
        })
    }
}

fn global_label(var: &naga::GlobalVariable, handle: Handle<naga::GlobalVariable>) -> String {
    var.name
        .clone()
        .unwrap_or_else(|| format!("<unnamed global #{}>", handle.index()))
}

fn type_label(ty: &Type, handle: Handle<Type>) -> String {
    ty.name
        .clone()
        .unwrap_or_else(|| format!("<type #{}>", handle.index()))
}

fn validate_type_handle(
    module: &naga::Module,
    handle: Handle<Type>,
    depth: usize,
    active: &mut BTreeSet<usize>,
    validated: &mut BTreeSet<usize>,
    context: &str,
) -> Result<(), LoweringError> {
    if depth > MAX_TYPE_DEPTH || active.contains(&handle.index()) {
        return Err(LoweringError::CyclicOrDeepType {
            type_index: handle.index(),
            max_depth: MAX_TYPE_DEPTH,
        });
    }
    if validated.contains(&handle.index()) {
        return Ok(());
    }

    let ty = ensure_type_handle(module, handle, context)?;
    let name = type_label(ty, handle);
    active.insert(handle.index());

    match &ty.inner {
        TypeInner::Struct { members, span } => {
            if members.is_empty() || *span == 0 {
                active.remove(&handle.index());
                return Err(LoweringError::ZeroSizedOrInvalidType {
                    type_name: name,
                    detail: format!(
                        "struct has {} member(s) and byte span {} (zero-sized structs are invalid in WGSL/WebGPU)",
                        members.len(),
                        span
                    ),
                });
            }

            let last_idx = members.len() - 1;
            for (idx, member) in members.iter().enumerate() {
                let member_ctx = format!("struct '{name}' member #{idx}");
                let member_ty = ensure_type_handle(module, member.ty, &member_ctx)?;
                let is_dynamic_array = matches!(
                    member_ty.inner,
                    TypeInner::Array {
                        size: ArraySize::Dynamic,
                        ..
                    }
                );

                if is_dynamic_array && idx != last_idx {
                    active.remove(&handle.index());
                    return Err(LoweringError::ZeroSizedOrInvalidType {
                        type_name: name,
                        detail: format!(
                            "runtime-sized array at member #{idx} must be the final member of the struct"
                        ),
                    });
                }

                if member.offset > *span || (!is_dynamic_array && member.offset == *span) {
                    active.remove(&handle.index());
                    return Err(LoweringError::ZeroSizedOrInvalidType {
                        type_name: name,
                        detail: format!(
                            "member #{idx} offset {} exceeds struct span {}",
                            member.offset, span
                        ),
                    });
                }

                validate_type_handle(module, member.ty, depth + 1, active, validated, &member_ctx)?;
            }
        }
        TypeInner::Array { base, size, stride } => {
            if *stride == 0 {
                active.remove(&handle.index());
                return Err(LoweringError::ZeroSizedOrInvalidType {
                    type_name: name,
                    detail: "array stride is 0 (zero-sized/zero-stride arrays are invalid)"
                        .to_string(),
                });
            }
            match *size {
                ArraySize::Constant(count) => {
                    if stride.checked_mul(count.get()).is_none() {
                        active.remove(&handle.index());
                        return Err(LoweringError::ZeroSizedOrInvalidType {
                            type_name: name,
                            detail: format!(
                                "array byte size overflows u32 (stride {stride} * count {})",
                                count.get()
                            ),
                        });
                    }
                }
                ArraySize::Pending(override_handle) => {
                    if override_handle.index() >= module.overrides.len() {
                        active.remove(&handle.index());
                        return Err(LoweringError::InvalidHandle {
                            arena: "overrides",
                            index: override_handle.index(),
                            len: module.overrides.len(),
                            context: format!("array '{name}' pending size override"),
                        });
                    }
                }
                ArraySize::Dynamic => {}
            }
            validate_type_handle(
                module,
                *base,
                depth + 1,
                active,
                validated,
                &format!("array '{name}' element type"),
            )?;
        }
        TypeInner::BindingArray { base, size } => {
            if let ArraySize::Pending(override_handle) = *size
                && override_handle.index() >= module.overrides.len()
            {
                active.remove(&handle.index());
                return Err(LoweringError::InvalidHandle {
                    arena: "overrides",
                    index: override_handle.index(),
                    len: module.overrides.len(),
                    context: format!("binding_array '{name}' pending size override"),
                });
            }
            validate_type_handle(
                module,
                *base,
                depth + 1,
                active,
                validated,
                &format!("binding_array '{name}' element type"),
            )?;
        }
        TypeInner::Pointer { base, .. } => {
            validate_type_handle(
                module,
                *base,
                depth + 1,
                active,
                validated,
                &format!("pointer '{name}' base type"),
            )?;
        }
        TypeInner::Atomic(scalar) => {
            if !is_baseline_atomic_scalar(*scalar) {
                active.remove(&handle.index());
                return Err(LoweringError::InvalidAtomicOp {
                    context: format!("type '{name}'"),
                    detail: format!(
                        "atomic scalar {:?} ({}-bit) is not supported in baseline WebGPU (only 32-bit i32/u32)",
                        scalar.kind,
                        u32::from(scalar.width) * 8
                    ),
                });
            }
        }
        _ => {}
    }

    active.remove(&handle.index());
    validated.insert(handle.index());
    Ok(())
}

fn is_baseline_atomic_scalar(scalar: naga::Scalar) -> bool {
    matches!(scalar.kind, ScalarKind::Sint | ScalarKind::Uint) && scalar.width == 4
}

fn type_contains_atomic(module: &naga::Module, handle: Handle<Type>, depth: usize) -> bool {
    if depth > MAX_TYPE_DEPTH {
        return false;
    }
    let Some(ty) = lookup_type(module, handle) else {
        return false;
    };
    match &ty.inner {
        TypeInner::Atomic(_) => true,
        TypeInner::Array { base, .. }
        | TypeInner::BindingArray { base, .. }
        | TypeInner::Pointer { base, .. } => type_contains_atomic(module, *base, depth + 1),
        TypeInner::Struct { members, .. } => members
            .iter()
            .any(|m| type_contains_atomic(module, m.ty, depth + 1)),
        _ => false,
    }
}

fn type_contains_dynamic_array(module: &naga::Module, handle: Handle<Type>, depth: usize) -> bool {
    if depth > MAX_TYPE_DEPTH {
        return false;
    }
    let Some(ty) = lookup_type(module, handle) else {
        return false;
    };
    match &ty.inner {
        TypeInner::Array {
            size: ArraySize::Dynamic,
            ..
        } => true,
        TypeInner::Array { base, .. } | TypeInner::Pointer { base, .. } => {
            type_contains_dynamic_array(module, *base, depth + 1)
        }
        TypeInner::Struct { members, .. } => members
            .iter()
            .any(|m| type_contains_dynamic_array(module, m.ty, depth + 1)),
        _ => false,
    }
}

fn type_contains_opaque_or_pointer(
    module: &naga::Module,
    handle: Handle<Type>,
    depth: usize,
) -> Option<&'static str> {
    if depth > MAX_TYPE_DEPTH {
        return Some("type exceeds maximum nesting depth");
    }
    let ty = lookup_type(module, handle)?;
    match &ty.inner {
        TypeInner::Image { .. } => Some("image/texture handle"),
        TypeInner::Sampler { .. } => Some("sampler handle"),
        TypeInner::AccelerationStructure { .. } => Some("acceleration structure"),
        TypeInner::RayQuery { .. } => Some("ray query"),
        TypeInner::BindingArray { .. } => Some("binding_array"),
        TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => Some("pointer type"),
        TypeInner::Array { base, .. } => type_contains_opaque_or_pointer(module, *base, depth + 1),
        TypeInner::Struct { members, .. } => members
            .iter()
            .find_map(|m| type_contains_opaque_or_pointer(module, m.ty, depth + 1)),
        _ => None,
    }
}

fn validate_push_constant_global(
    module: &naga::Module,
    handle: Handle<naga::GlobalVariable>,
    var: &naga::GlobalVariable,
) -> Result<(), LoweringError> {
    let label = global_label(var, handle);
    let mut active = BTreeSet::new();
    let mut validated = BTreeSet::new();
    validate_type_handle(
        module,
        var.ty,
        0,
        &mut active,
        &mut validated,
        &format!("push-constant global '{label}'"),
    )?;

    if var.init.is_some() {
        return Err(LoweringError::UnsupportedPushConstantType {
            variable: label,
            detail: "push-constant global cannot have an initializer when lowered to var<uniform>"
                .to_string(),
        });
    }
    if type_contains_atomic(module, var.ty, 0) {
        return Err(LoweringError::UnsupportedPushConstantType {
            variable: label,
            detail: "push-constant type contains an atomic, which is illegal in var<uniform>"
                .to_string(),
        });
    }
    if type_contains_dynamic_array(module, var.ty, 0) {
        return Err(LoweringError::UnsupportedPushConstantType {
            variable: label,
            detail: "push-constant type contains a runtime-sized array, which is illegal in var<uniform>"
                .to_string(),
        });
    }
    if let Some(opaque_kind) = type_contains_opaque_or_pointer(module, var.ty, 0) {
        return Err(LoweringError::UnsupportedPushConstantType {
            variable: label,
            detail: format!("push-constant type contains {opaque_kind}, which is illegal in var<uniform>"),
        });
    }

    Ok(())
}

fn validate_global_variable(
    module: &naga::Module,
    handle: Handle<naga::GlobalVariable>,
    var: &naga::GlobalVariable,
) -> Result<(), LoweringError> {
    let label = global_label(var, handle);
    let var_ty = ensure_type_handle(module, var.ty, &format!("global variable '{label}'"))?;
    if let Some(init) = var.init {
        ensure_global_expr_handle(module, init, &format!("global variable '{label}' init"))?;
    }

    match var.space {
        AddressSpace::Immediate => {
            validate_push_constant_global(module, handle, var)?;
        }
        AddressSpace::Uniform => {
            if var.binding.is_none() {
                return Err(LoweringError::InvalidResourceBinding {
                    variable: label,
                    detail: "uniform buffer global is missing @group/@binding".to_string(),
                });
            }
            if type_contains_atomic(module, var.ty, 0) {
                return Err(LoweringError::InvalidAtomicOp {
                    context: format!("uniform global '{label}'"),
                    detail: "atomics are forbidden in uniform address space".to_string(),
                });
            }
            if type_contains_dynamic_array(module, var.ty, 0) {
                return Err(LoweringError::InvalidStorageOp {
                    context: format!("uniform global '{label}'"),
                    detail: "runtime-sized arrays are forbidden in uniform address space"
                        .to_string(),
                });
            }
        }
        AddressSpace::Storage { access } => {
            if var.binding.is_none() {
                return Err(LoweringError::InvalidResourceBinding {
                    variable: label,
                    detail: "storage buffer global is missing @group/@binding".to_string(),
                });
            }
            if access.is_empty() {
                return Err(LoweringError::InvalidStorageOp {
                    context: format!("storage global '{label}'"),
                    detail: "storage buffer has empty StorageAccess flags".to_string(),
                });
            }
            if type_contains_atomic(module, var.ty, 0) && !access.contains(StorageAccess::STORE) {
                return Err(LoweringError::InvalidAtomicOp {
                    context: format!("storage global '{label}'"),
                    detail: "atomic storage buffer requires read_write (STORE) access".to_string(),
                });
            }
        }
        AddressSpace::Handle => {
            if var.binding.is_none() {
                return Err(LoweringError::InvalidResourceBinding {
                    variable: label,
                    detail: "handle resource global (texture/sampler) is missing @group/@binding"
                        .to_string(),
                });
            }
            let inner = match var_ty.inner {
                TypeInner::BindingArray { base, .. } => {
                    &ensure_type_handle(module, base, &format!("binding_array global '{label}'"))?
                        .inner
                }
                _ => &var_ty.inner,
            };
            if let TypeInner::Image {
                dim,
                arrayed,
                class,
            } = *inner
            {
                validate_image_type(&label, dim, arrayed, class)?;
            }
        }
        AddressSpace::Private | AddressSpace::WorkGroup | AddressSpace::Function => {
            if var.binding.is_some() {
                return Err(LoweringError::InvalidResourceBinding {
                    variable: label,
                    detail: format!(
                        "{:?} global must not declare @group/@binding",
                        var.space
                    ),
                });
            }
            if matches!(var.space, AddressSpace::Private | AddressSpace::Function)
                && type_contains_atomic(module, var.ty, 0)
            {
                return Err(LoweringError::InvalidAtomicOp {
                    context: format!("global '{label}'"),
                    detail: format!(
                        "atomics are forbidden in {:?} address space (only storage/workgroup)",
                        var.space
                    ),
                });
            }
        }
        _ => {}
    }

    Ok(())
}

fn validate_image_type(
    label: &str,
    dim: ImageDimension,
    arrayed: bool,
    class: ImageClass,
) -> Result<(), LoweringError> {
    if arrayed && matches!(dim, ImageDimension::D1 | ImageDimension::D3) {
        return Err(LoweringError::InvalidTextureOp {
            context: format!("texture global '{label}'"),
            detail: format!("arrayed {dim:?} textures are not supported in WebGPU"),
        });
    }

    match class {
        ImageClass::Sampled { multi: true, .. } | ImageClass::Depth { multi: true } => {
            if dim != ImageDimension::D2 || arrayed {
                return Err(LoweringError::InvalidTextureOp {
                    context: format!("texture global '{label}'"),
                    detail: format!(
                        "multisampled textures must be non-arrayed 2D (got dim={dim:?}, arrayed={arrayed})"
                    ),
                });
            }
        }
        ImageClass::Depth { multi: false } => {
            if matches!(dim, ImageDimension::D1 | ImageDimension::D3) {
                return Err(LoweringError::InvalidTextureOp {
                    context: format!("texture global '{label}'"),
                    detail: format!("depth textures cannot have dimension {dim:?} in WebGPU"),
                });
            }
        }
        ImageClass::Storage { format, access } => {
            if access.is_empty() {
                return Err(LoweringError::InvalidTextureOp {
                    context: format!("storage texture global '{label}'"),
                    detail: "storage texture has empty StorageAccess flags".to_string(),
                });
            }
            if access.contains(StorageAccess::ATOMIC) {
                return Err(LoweringError::InvalidAtomicOp {
                    context: format!("storage texture global '{label}'"),
                    detail: "atomic storage textures are not supported in baseline WebGPU"
                        .to_string(),
                });
            }
            if dim == ImageDimension::Cube {
                return Err(LoweringError::InvalidTextureOp {
                    context: format!("storage texture global '{label}'"),
                    detail: "cube storage textures are not supported in WebGPU".to_string(),
                });
            }
            if access.contains(StorageAccess::LOAD)
                && access.contains(StorageAccess::STORE)
                && !matches!(
                    format,
                    StorageFormat::R32Uint | StorageFormat::R32Sint | StorageFormat::R32Float
                )
            {
                return Err(LoweringError::InvalidTextureOp {
                    context: format!("storage texture global '{label}'"),
                    detail: format!(
                        "read_write storage texture format {format:?} is not supported in WebGPU core (only r32uint/r32sint/r32float)"
                    ),
                });
            }
        }
        ImageClass::Sampled { multi: false, .. } | ImageClass::External => {}
    }

    Ok(())
}

fn validate_binding_collisions(module: &naga::Module) -> Result<(), LoweringError> {
    if module.entry_points.is_empty() {
        let all_globals: BTreeSet<Handle<naga::GlobalVariable>> =
            module.global_variables.iter().map(|(h, _)| h).collect();
        return check_binding_set_collisions(module, &all_globals);
    }

    let call_graph = build_call_graph(module)?;
    for ep_idx in 0..module.entry_points.len() {
        let reachable = collect_reachable_functions(&call_graph, FnRef::Entry(ep_idx));
        let mut used_globals = BTreeSet::new();
        for fn_ref in reachable {
            let func = match fn_ref {
                FnRef::Entry(i) => &module.entry_points[i].function,
                FnRef::Function(h) => {
                    if h.index() >= module.functions.len() {
                        continue;
                    }
                    &module.functions[h]
                }
            };
            for (_, expr) in func.expressions.iter() {
                if let Expression::GlobalVariable(g) = *expr {
                    used_globals.insert(g);
                }
            }
        }
        check_binding_set_collisions(module, &used_globals)?;
    }

    Ok(())
}

fn check_binding_set_collisions(
    module: &naga::Module,
    globals: &BTreeSet<Handle<naga::GlobalVariable>>,
) -> Result<(), LoweringError> {
    let mut seen: BTreeMap<(u32, u32), (Handle<naga::GlobalVariable>, String)> = BTreeMap::new();
    for &handle in globals {
        let Some(var) = lookup_global(module, handle) else {
            continue;
        };
        if var.space == AddressSpace::Immediate {
            continue;
        }
        let Some(binding) = var.binding else {
            continue;
        };
        let key = (binding.group, binding.binding);
        let label = global_label(var, handle);
        if let Some((first_handle, first_label)) = seen.get(&key)
            && *first_handle != handle
        {
            return Err(LoweringError::DuplicateResourceBinding {
                group: binding.group,
                binding: binding.binding,
                first_variable: first_label.clone(),
                second_variable: label,
            });
        }
        seen.insert(key, (handle, label));
    }
    Ok(())
}

fn build_call_graph(module: &naga::Module) -> Result<BTreeMap<FnRef, BTreeSet<FnRef>>, LoweringError> {
    let mut graph: BTreeMap<FnRef, BTreeSet<FnRef>> = BTreeMap::new();

    for (fn_handle, func) in module.functions.iter() {
        let fn_name = func
            .name
            .clone()
            .unwrap_or_else(|| format!("function#{}", fn_handle.index()));
        let mut callees = BTreeSet::new();
        collect_block_callees(module, &func.body, &fn_name, 0, &mut callees)?;
        graph.insert(FnRef::Function(fn_handle), callees);
    }

    for (ep_idx, ep) in module.entry_points.iter().enumerate() {
        let fn_name = format!("entry_point '{}'", ep.name);
        let mut callees = BTreeSet::new();
        collect_block_callees(module, &ep.function.body, &fn_name, 0, &mut callees)?;
        graph.insert(FnRef::Entry(ep_idx), callees);
    }

    Ok(graph)
}

fn collect_block_callees(
    module: &naga::Module,
    block: &Block,
    fn_name: &str,
    depth: usize,
    callees: &mut BTreeSet<FnRef>,
) -> Result<(), LoweringError> {
    if depth > MAX_BLOCK_DEPTH {
        return Err(LoweringError::BlockDepthExceeded {
            function: fn_name.to_string(),
            max_depth: MAX_BLOCK_DEPTH,
        });
    }
    for stmt in block.iter() {
        match stmt {
            Statement::Call { function, .. } => {
                ensure_function_handle(module, *function, fn_name)?;
                callees.insert(FnRef::Function(*function));
            }
            Statement::Block(inner) => {
                collect_block_callees(module, inner, fn_name, depth + 1, callees)?;
            }
            Statement::If { accept, reject, .. } => {
                collect_block_callees(module, accept, fn_name, depth + 1, callees)?;
                collect_block_callees(module, reject, fn_name, depth + 1, callees)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_block_callees(module, &case.body, fn_name, depth + 1, callees)?;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_block_callees(module, body, fn_name, depth + 1, callees)?;
                collect_block_callees(module, continuing, fn_name, depth + 1, callees)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_reachable_functions(
    graph: &BTreeMap<FnRef, BTreeSet<FnRef>>,
    root: FnRef,
) -> BTreeSet<FnRef> {
    let mut visited = BTreeSet::new();
    let mut worklist = vec![root];
    while let Some(node) = worklist.pop() {
        if !visited.insert(node) {
            continue;
        }
        if let Some(callees) = graph.get(&node) {
            for &callee in callees {
                if !visited.contains(&callee) {
                    worklist.push(callee);
                }
            }
        }
    }
    visited
}

fn compute_function_stage_reachability(
    module: &naga::Module,
) -> Result<BTreeMap<FnRef, BTreeSet<ShaderStage>>, LoweringError> {
    let graph = build_call_graph(module)?;
    let mut stages: BTreeMap<FnRef, BTreeSet<ShaderStage>> = BTreeMap::new();
    for (ep_idx, ep) in module.entry_points.iter().enumerate() {
        for fn_ref in collect_reachable_functions(&graph, FnRef::Entry(ep_idx)) {
            stages.entry(fn_ref).or_default().insert(ep.stage);
        }
    }
    Ok(stages)
}

fn validate_function(
    module: &naga::Module,
    func: &Function,
    fn_name: &str,
    stages: &BTreeSet<ShaderStage>,
) -> Result<(), LoweringError> {
    for (idx, arg) in func.arguments.iter().enumerate() {
        ensure_type_handle(module, arg.ty, &format!("{fn_name} arg #{idx}"))?;
    }
    if let Some(result) = &func.result {
        ensure_type_handle(module, result.ty, &format!("{fn_name} return type"))?;
    }
    for (lv_handle, lv) in func.local_variables.iter() {
        let lv_ctx = format!(
            "{fn_name} local '{}'",
            lv.name.as_deref().unwrap_or("<unnamed>")
        );
        ensure_type_handle(module, lv.ty, &lv_ctx)?;
        if type_contains_atomic(module, lv.ty, 0) {
            return Err(LoweringError::InvalidAtomicOp {
                context: lv_ctx,
                detail: "atomics are forbidden in function-local variables".to_string(),
            });
        }
        if let Some(init) = lv.init {
            ensure_func_expr_handle(func, init, &lv_ctx)?;
        }
        let _ = lv_handle;
    }

    let reachable_from_non_fragment = stages
        .iter()
        .any(|&s| !matches!(s, ShaderStage::Fragment));

    // Validate all expressions in deterministic arena index order.
    for (expr_handle, expr) in func.expressions.iter() {
        let expr_ctx = format!("{fn_name} expr #{}", expr_handle.index());
        match expr {
            Expression::Constant(c) => {
                if c.index() >= module.constants.len() {
                    return Err(LoweringError::InvalidHandle {
                        arena: "constants",
                        index: c.index(),
                        len: module.constants.len(),
                        context: expr_ctx,
                    });
                }
            }
            Expression::Override(o) => {
                if o.index() >= module.overrides.len() {
                    return Err(LoweringError::InvalidHandle {
                        arena: "overrides",
                        index: o.index(),
                        len: module.overrides.len(),
                        context: expr_ctx,
                    });
                }
            }
            Expression::ZeroValue(ty) => {
                ensure_type_handle(module, *ty, &expr_ctx)?;
            }
            Expression::Compose { ty, components } => {
                ensure_type_handle(module, *ty, &expr_ctx)?;
                for &comp in components {
                    ensure_func_expr_handle(func, comp, &expr_ctx)?;
                }
            }
            Expression::Access { base, index } => {
                ensure_func_expr_handle(func, *base, &expr_ctx)?;
                ensure_func_expr_handle(func, *index, &expr_ctx)?;
            }
            Expression::AccessIndex { base, .. } => {
                ensure_func_expr_handle(func, *base, &expr_ctx)?;
            }
            Expression::Splat { value, .. } => {
                ensure_func_expr_handle(func, *value, &expr_ctx)?;
            }
            Expression::Swizzle { vector, .. } => {
                ensure_func_expr_handle(func, *vector, &expr_ctx)?;
            }
            Expression::FunctionArgument(idx) => {
                if (*idx as usize) >= func.arguments.len() {
                    return Err(LoweringError::InvalidHandle {
                        arena: "function_arguments",
                        index: *idx as usize,
                        len: func.arguments.len(),
                        context: expr_ctx,
                    });
                }
            }
            Expression::GlobalVariable(g) => {
                ensure_global_handle(module, *g, &expr_ctx)?;
            }
            Expression::LocalVariable(lv) => {
                if lv.index() >= func.local_variables.len() {
                    return Err(LoweringError::InvalidHandle {
                        arena: "local_variables",
                        index: lv.index(),
                        len: func.local_variables.len(),
                        context: expr_ctx,
                    });
                }
            }
            Expression::Load { pointer } => {
                ensure_func_expr_handle(func, *pointer, &expr_ctx)?;
            }
            Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                ..
            } => {
                ensure_func_expr_handle(func, *image, &expr_ctx)?;
                ensure_func_expr_handle(func, *sampler, &expr_ctx)?;
                ensure_func_expr_handle(func, *coordinate, &expr_ctx)?;
                if let Some(ai) = *array_index {
                    ensure_func_expr_handle(func, ai, &expr_ctx)?;
                }
                if let Some(off) = *offset {
                    ensure_func_expr_handle(func, off, &expr_ctx)?;
                }
                if let Some(dr) = *depth_ref {
                    ensure_func_expr_handle(func, dr, &expr_ctx)?;
                }
                match *level {
                    SampleLevel::Auto | SampleLevel::Zero => {}
                    SampleLevel::Exact(h) | SampleLevel::Bias(h) => {
                        ensure_func_expr_handle(func, h, &expr_ctx)?;
                    }
                    SampleLevel::Gradient { x, y } => {
                        ensure_func_expr_handle(func, x, &expr_ctx)?;
                        ensure_func_expr_handle(func, y, &expr_ctx)?;
                    }
                }
                // `textureGather` (`gather.is_some()`) does not use implicit derivatives and is
                // valid in all stages; `textureSample` (`Auto`) and `textureSampleBias` (`Bias`)
                // require implicit derivatives and are Fragment-only in WebGPU/WGSL.
                if gather.is_none()
                    && matches!(level, SampleLevel::Auto | SampleLevel::Bias(_))
                    && reachable_from_non_fragment
                {
                    return Err(LoweringError::InvalidTextureOp {
                        context: expr_ctx,
                        detail: "implicit-derivative textureSample/textureSampleBias is only valid in fragment shaders; use textureSampleLevel or textureSampleGrad in vertex/compute stages".to_string(),
                    });
                }
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                ensure_func_expr_handle(func, *image, &expr_ctx)?;
                ensure_func_expr_handle(func, *coordinate, &expr_ctx)?;
                if let Some(ai) = *array_index {
                    ensure_func_expr_handle(func, ai, &expr_ctx)?;
                }
                if let Some(s) = *sample {
                    ensure_func_expr_handle(func, s, &expr_ctx)?;
                }
                if let Some(l) = *level {
                    ensure_func_expr_handle(func, l, &expr_ctx)?;
                }
            }
            Expression::ImageQuery { image, query } => {
                ensure_func_expr_handle(func, *image, &expr_ctx)?;
                if let ImageQuery::Size { level: Some(l) } = *query {
                    ensure_func_expr_handle(func, l, &expr_ctx)?;
                }
            }
            Expression::Unary { expr: inner, .. } => {
                ensure_func_expr_handle(func, *inner, &expr_ctx)?;
            }
            Expression::Binary { left, right, .. } => {
                ensure_func_expr_handle(func, *left, &expr_ctx)?;
                ensure_func_expr_handle(func, *right, &expr_ctx)?;
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                ensure_func_expr_handle(func, *condition, &expr_ctx)?;
                ensure_func_expr_handle(func, *accept, &expr_ctx)?;
                ensure_func_expr_handle(func, *reject, &expr_ctx)?;
            }
            Expression::Derivative { expr: inner, .. } => {
                ensure_func_expr_handle(func, *inner, &expr_ctx)?;
                if reachable_from_non_fragment {
                    return Err(LoweringError::InvalidDerivativeOp {
                        context: expr_ctx,
                        detail: "derivative operations (dpdx/dpdy/fwidth) are only valid in fragment shaders".to_string(),
                    });
                }
            }
            Expression::Relational { argument, .. } => {
                ensure_func_expr_handle(func, *argument, &expr_ctx)?;
            }
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                ensure_func_expr_handle(func, *arg, &expr_ctx)?;
                if let Some(h) = *arg1 {
                    ensure_func_expr_handle(func, h, &expr_ctx)?;
                }
                if let Some(h) = *arg2 {
                    ensure_func_expr_handle(func, h, &expr_ctx)?;
                }
                if let Some(h) = *arg3 {
                    ensure_func_expr_handle(func, h, &expr_ctx)?;
                }
            }
            Expression::As { expr: inner, .. } => {
                ensure_func_expr_handle(func, *inner, &expr_ctx)?;
            }
            Expression::CallResult(callee) => {
                ensure_function_handle(module, *callee, &expr_ctx)?;
            }
            Expression::AtomicResult { ty, comparison } => {
                let res_ty = ensure_type_handle(module, *ty, &expr_ctx)?;
                validate_atomic_result_type(module, res_ty, *comparison, &expr_ctx)?;
            }
            Expression::WorkGroupUniformLoadResult { ty } => {
                ensure_type_handle(module, *ty, &expr_ctx)?;
            }
            Expression::ArrayLength(inner) => {
                ensure_func_expr_handle(func, *inner, &expr_ctx)?;
            }
            Expression::SubgroupOperationResult { ty } => {
                ensure_type_handle(module, *ty, &expr_ctx)?;
            }
            Expression::RayQueryGetIntersection { query, .. }
            | Expression::RayQueryVertexPositions { query, .. } => {
                ensure_func_expr_handle(func, *query, &expr_ctx)?;
            }
            Expression::Literal(_)
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult => {}
            _ => {}
        }
    }

    validate_block(module, func, &func.body, fn_name, 0)
}

fn validate_atomic_result_type(
    module: &naga::Module,
    res_ty: &Type,
    comparison: bool,
    context: &str,
) -> Result<(), LoweringError> {
    if !comparison {
        match res_ty.inner {
            TypeInner::Scalar(scalar) if is_baseline_atomic_scalar(scalar) => Ok(()),
            _ => Err(LoweringError::InvalidAtomicOp {
                context: context.to_string(),
                detail: format!(
                    "atomic result type {:?} is not a baseline WebGPU 32-bit integer scalar",
                    res_ty.inner
                ),
            }),
        }
    } else {
        match &res_ty.inner {
            TypeInner::Struct { members, .. } if members.len() == 2 => {
                let first_ty = ensure_type_handle(module, members[0].ty, context)?;
                let second_ty = ensure_type_handle(module, members[1].ty, context)?;
                let first_ok = matches!(
                    first_ty.inner,
                    TypeInner::Scalar(s) if is_baseline_atomic_scalar(s)
                );
                let second_ok = matches!(
                    second_ty.inner,
                    TypeInner::Scalar(s) if s.kind == ScalarKind::Bool
                );
                if first_ok && second_ok {
                    Ok(())
                } else {
                    Err(LoweringError::InvalidAtomicOp {
                        context: context.to_string(),
                        detail: "atomicCompareExchangeWeak result struct must have (i32|u32, bool) members".to_string(),
                    })
                }
            }
            _ => Err(LoweringError::InvalidAtomicOp {
                context: context.to_string(),
                detail: "atomicCompareExchangeWeak result must be a two-member struct".to_string(),
            }),
        }
    }
}

fn validate_block(
    module: &naga::Module,
    func: &Function,
    block: &Block,
    fn_name: &str,
    depth: usize,
) -> Result<(), LoweringError> {
    if depth > MAX_BLOCK_DEPTH {
        return Err(LoweringError::BlockDepthExceeded {
            function: fn_name.to_string(),
            max_depth: MAX_BLOCK_DEPTH,
        });
    }

    for stmt in block.iter() {
        match stmt {
            Statement::Emit(range) => {
                // Empty ranges (`first_and_last() == None`) are safely ignored (no panic!).
                if let Some((first, last)) = range.first_and_last() {
                    let start = first.index();
                    let end = last.index();
                    if start > end || end >= func.expressions.len() {
                        return Err(LoweringError::InvalidExpressionRange {
                            function: fn_name.to_string(),
                            start,
                            end,
                            expr_count: func.expressions.len(),
                        });
                    }
                }
            }
            Statement::Block(inner) => {
                validate_block(module, func, inner, fn_name, depth + 1)?;
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                ensure_func_expr_handle(func, *condition, fn_name)?;
                validate_block(module, func, accept, fn_name, depth + 1)?;
                validate_block(module, func, reject, fn_name, depth + 1)?;
            }
            Statement::Switch { selector, cases } => {
                ensure_func_expr_handle(func, *selector, fn_name)?;
                for case in cases {
                    validate_block(module, func, &case.body, fn_name, depth + 1)?;
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                if let Some(cond) = *break_if {
                    ensure_func_expr_handle(func, cond, fn_name)?;
                }
                validate_block(module, func, body, fn_name, depth + 1)?;
                validate_block(module, func, continuing, fn_name, depth + 1)?;
            }
            Statement::Return { value } => {
                if let Some(val) = *value {
                    ensure_func_expr_handle(func, val, fn_name)?;
                }
            }
            Statement::Store { pointer, value } => {
                ensure_func_expr_handle(func, *pointer, fn_name)?;
                ensure_func_expr_handle(func, *value, fn_name)?;
                if let Some(root_global) = trace_pointer_root_global(func, *pointer)
                    && let Some(var) = lookup_global(module, root_global)
                {
                    match var.space {
                        AddressSpace::Uniform | AddressSpace::Immediate => {
                            return Err(LoweringError::InvalidStorageOp {
                                context: fn_name.to_string(),
                                detail: format!(
                                    "cannot store into read-only {:?} global '{}'",
                                    var.space,
                                    global_label(var, root_global)
                                ),
                            });
                        }
                        AddressSpace::Storage { access }
                            if !access.contains(StorageAccess::STORE) =>
                        {
                            return Err(LoweringError::InvalidStorageOp {
                                context: fn_name.to_string(),
                                detail: format!(
                                    "cannot store into read-only storage buffer '{}'",
                                    global_label(var, root_global)
                                ),
                            });
                        }
                        _ => {}
                    }
                }
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                ensure_func_expr_handle(func, *image, fn_name)?;
                ensure_func_expr_handle(func, *coordinate, fn_name)?;
                if let Some(ai) = *array_index {
                    ensure_func_expr_handle(func, ai, fn_name)?;
                }
                ensure_func_expr_handle(func, *value, fn_name)?;
            }
            Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                ensure_func_expr_handle(func, *image, fn_name)?;
                ensure_func_expr_handle(func, *coordinate, fn_name)?;
                if let Some(ai) = *array_index {
                    ensure_func_expr_handle(func, ai, fn_name)?;
                }
                ensure_func_expr_handle(func, *value, fn_name)?;
                if let AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                    ensure_func_expr_handle(func, cmp, fn_name)?;
                }
                return Err(LoweringError::InvalidAtomicOp {
                    context: fn_name.to_string(),
                    detail: "Statement::ImageAtomic is not supported in baseline WebGPU"
                        .to_string(),
                });
            }
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                ensure_func_expr_handle(func, *pointer, fn_name)?;
                ensure_func_expr_handle(func, *value, fn_name)?;
                if let Some(res) = *result {
                    ensure_func_expr_handle(func, res, fn_name)?;
                }
                if let AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                    ensure_func_expr_handle(func, cmp, fn_name)?;
                }
                if let Some(root_global) = trace_pointer_root_global(func, *pointer)
                    && let Some(var) = lookup_global(module, root_global)
                {
                    let valid_space = match var.space {
                        AddressSpace::WorkGroup => true,
                        AddressSpace::Storage { access } => access.contains(StorageAccess::STORE),
                        _ => false,
                    };
                    if !valid_space {
                        return Err(LoweringError::InvalidAtomicOp {
                            context: fn_name.to_string(),
                            detail: format!(
                                "atomic operation on '{}' requires read_write storage or workgroup address space (got {:?})",
                                global_label(var, root_global),
                                var.space
                            ),
                        });
                    }
                }
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                ensure_func_expr_handle(func, *pointer, fn_name)?;
                ensure_func_expr_handle(func, *result, fn_name)?;
            }
            Statement::Call {
                function,
                arguments,
                result,
            } => {
                ensure_function_handle(module, *function, fn_name)?;
                for &arg in arguments {
                    ensure_func_expr_handle(func, arg, fn_name)?;
                }
                if let Some(res) = *result {
                    ensure_func_expr_handle(func, res, fn_name)?;
                }
            }
            Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => {}
            _ => {}
        }
    }

    Ok(())
}

fn trace_pointer_root_global(
    func: &Function,
    mut expr: Handle<Expression>,
) -> Option<Handle<naga::GlobalVariable>> {
    for _ in 0..MAX_POINTER_TRACE_DEPTH {
        match lookup_expr(func, expr)? {
            Expression::GlobalVariable(g) => return Some(*g),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                expr = *base;
            }
            _ => return None,
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Deterministic binding collection, selection & pointer-type rewriting
// ---------------------------------------------------------------------------

fn collect_used_bindings_in_webgpu_groups(
    module: &naga::Module,
) -> [BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize] {
    let mut used: [BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize] = Default::default();
    for (_, var) in module.global_variables.iter() {
        if var.space == AddressSpace::Immediate {
            continue;
        }
        if let Some(binding) = var.binding
            && binding.group < WEBGPU_MAX_BIND_GROUPS
        {
            used[binding.group as usize].insert(binding.binding);
        }
    }
    used
}

fn collect_sorted_immediate_handles(module: &naga::Module) -> Vec<Handle<naga::GlobalVariable>> {
    let mut items: Vec<(bool, &str, usize, Handle<naga::GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            if var.space == AddressSpace::Immediate {
                Some((
                    var.name.is_none(),
                    var.name.as_deref().unwrap_or(""),
                    handle.index(),
                    handle,
                ))
            } else {
                None
            }
        })
        .collect();
    items.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    items.into_iter().map(|(_, _, _, h)| h).collect()
}

fn collect_sorted_overflow_bindings(
    module: &naga::Module,
) -> Vec<(u32, u32, Handle<naga::GlobalVariable>)> {
    let mut items: Vec<(u32, u32, bool, &str, usize, Handle<naga::GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            if var.space == AddressSpace::Immediate {
                return None;
            }
            let binding = var.binding?;
            if binding.group >= WEBGPU_MAX_BIND_GROUPS {
                Some((
                    binding.group,
                    binding.binding,
                    var.name.is_none(),
                    var.name.as_deref().unwrap_or(""),
                    handle.index(),
                    handle,
                ))
            } else {
                None
            }
        })
        .collect();
    items.sort_unstable_by(|a, b| {
        (a.0, a.1, a.2, a.3, a.4).cmp(&(b.0, b.1, b.2, b.3, b.4))
    });
    items.into_iter().map(|(g, b, _, _, _, h)| (g, b, h)).collect()
}

fn select_push_constant_group(used: &[BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize]) -> u32 {
    let preferred = PUSH_CONSTANT_TARGET_GROUP as usize;
    if used[preferred].len() < WEBGPU_MAX_BINDINGS_PER_GROUP {
        return PUSH_CONSTANT_TARGET_GROUP;
    }

    // Group 3 is at/above WebGPU's baseline per-group binding cap: pick the lowest group < 4 with space.
    for (g, occupied) in used.iter().enumerate().take(preferred) {
        if occupied.len() < WEBGPU_MAX_BINDINGS_PER_GROUP {
            return g as u32;
        }
    }

    // Fallback if every group is above the baseline cap: pick the least-populated group 0..4.
    select_least_populated_group(used)
}

fn select_least_populated_group(used: &[BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize]) -> u32 {
    (0..(WEBGPU_MAX_BIND_GROUPS as usize))
        .min_by_key(|&g| (used[g].len(), g))
        .unwrap_or(0) as u32
}

fn allocate_next_binding_checked(occupied: &mut BTreeSet<u32>) -> Option<u32> {
    let candidate = match occupied.last().copied() {
        None => 0,
        Some(max_existing) => match max_existing.checked_add(1) {
            Some(next) => next,
            None => {
                // `u32::MAX` is already taken; scan upward from 0 without wrapping.
                let mut slot = 0u32;
                loop {
                    if !occupied.contains(&slot) {
                        break slot;
                    }
                    slot = slot.checked_add(1)?;
                }
            }
        },
    };
    occupied.insert(candidate);
    Some(candidate)
}

fn rewrite_immediate_pointer_types(module: &mut naga::Module) {
    let candidates: Vec<(Handle<Type>, Type)> = module
        .types
        .iter()
        .filter_map(|(handle, ty)| match ty.inner {
            TypeInner::Pointer {
                base,
                space: AddressSpace::Immediate,
            } => Some((
                handle,
                Type {
                    name: ty.name.clone(),
                    inner: TypeInner::Pointer {
                        base,
                        space: AddressSpace::Uniform,
                    },
                },
            )),
            TypeInner::ValuePointer {
                size,
                scalar,
                space: AddressSpace::Immediate,
            } => Some((
                handle,
                Type {
                    name: ty.name.clone(),
                    inner: TypeInner::ValuePointer {
                        size,
                        scalar,
                        space: AddressSpace::Uniform,
                    },
                },
            )),
            _ => None,
        })
        .collect();

    if candidates.is_empty() {
        return;
    }

    let mut remapped_types: BTreeMap<Handle<Type>, Handle<Type>> = BTreeMap::new();
    for (handle, replacement) in candidates {
        if let Some((existing_handle, _)) = module
            .types
            .iter()
            .find(|(_, existing)| **existing == replacement)
        {
            // An identical `Uniform` pointer type already exists; record remap instead of
            // panicking in `UniqueArena::replace`.
            if existing_handle != handle {
                remapped_types.insert(handle, existing_handle);
            }
        } else {
            module.types.replace(handle, replacement);
        }
    }

    if !remapped_types.is_empty() {
        remap_type_handles_in_module(module, &remapped_types);
    }
}

fn remap_type_handle(ty: &mut Handle<Type>, remap: &BTreeMap<Handle<Type>, Handle<Type>>) {
    if let Some(&target) = remap.get(ty) {
        *ty = target;
    }
}

fn remap_type_handles_in_module(
    module: &mut naga::Module,
    remap: &BTreeMap<Handle<Type>, Handle<Type>>,
) {
    for (_, var) in module.global_variables.iter_mut() {
        remap_type_handle(&mut var.ty, remap);
    }
    for (_, c) in module.constants.iter_mut() {
        remap_type_handle(&mut c.ty, remap);
    }
    for (_, ov) in module.overrides.iter_mut() {
        remap_type_handle(&mut ov.ty, remap);
    }
    for (_, func) in module.functions.iter_mut() {
        remap_type_handles_in_function(func, remap);
    }
    for ep in &mut module.entry_points {
        remap_type_handles_in_function(&mut ep.function, remap);
    }
}

fn remap_type_handles_in_function(
    func: &mut Function,
    remap: &BTreeMap<Handle<Type>, Handle<Type>>,
) {
    for arg in &mut func.arguments {
        remap_type_handle(&mut arg.ty, remap);
    }
    if let Some(result) = &mut func.result {
        remap_type_handle(&mut result.ty, remap);
    }
    for (_, lv) in func.local_variables.iter_mut() {
        remap_type_handle(&mut lv.ty, remap);
    }
    for (_, expr) in func.expressions.iter_mut() {
        match expr {
            Expression::ZeroValue(ty)
            | Expression::Compose { ty, .. }
            | Expression::AtomicResult { ty, .. }
            | Expression::WorkGroupUniformLoadResult { ty }
            | Expression::SubgroupOperationResult { ty } => {
                remap_type_handle(ty, remap);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::back::wgsl;
    use naga::front::wgsl as wgsl_in;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::{
        Arena, Block, DerivativeAxis, DerivativeControl, EntryPoint, Expression, Function,
        FunctionArgument, GlobalVariable, MemoryDecorations, Module, Scalar, ShaderStage, Span,
        Statement, StorageAccess, StructMember, VectorSize,
    };
    use std::num::NonZeroU32;

    fn build_synthetic_module_with_push_constants_and_high_group() -> Module {
        let mut module = Module::default();

        let vec4_f32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::UNDEFINED,
        );

        let pc_struct_ty = module.types.insert(
            Type {
                name: Some("RootConstants".to_string()),
                inner: TypeInner::Struct {
                    members: vec![StructMember {
                        name: Some("tint".to_string()),
                        ty: vec4_f32_ty,
                        binding: None,
                        offset: 0,
                    }],
                    span: 16,
                },
            },
            Span::UNDEFINED,
        );

        let buf_struct_ty = module.types.insert(
            Type {
                name: Some("OutBuffer".to_string()),
                inner: TypeInner::Struct {
                    members: vec![StructMember {
                        name: Some("value".to_string()),
                        ty: vec4_f32_ty,
                        binding: None,
                        offset: 0,
                    }],
                    span: 16,
                },
            },
            Span::UNDEFINED,
        );

        // Existing binding in group 0 @binding(0)
        let g0_var = module.global_variables.append(
            GlobalVariable {
                name: Some("base_uniform".to_string()),
                space: AddressSpace::Uniform,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: pc_struct_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        // Push constant (AddressSpace::Immediate, no binding)
        let pc_var = module.global_variables.append(
            GlobalVariable {
                name: Some("push_data".to_string()),
                space: AddressSpace::Immediate,
                binding: None,
                ty: pc_struct_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        // Storage buffer in group 5 @binding(2) (exceeds WebGPU maxBindGroups = 4)
        let high_group_var = module.global_variables.append(
            GlobalVariable {
                name: Some("high_group_out".to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD | StorageAccess::STORE,
                },
                binding: Some(ResourceBinding {
                    group: 5,
                    binding: 2,
                }),
                ty: buf_struct_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        // Compute entry point that loads from g0_var and pc_var and stores into high_group_var.
        let mut func = Function {
            name: Some("main".to_string()),
            ..Function::default()
        };
        let g0_ptr = func
            .expressions
            .append(Expression::GlobalVariable(g0_var), Span::UNDEFINED);
        let g0_field = func.expressions.append(
            Expression::AccessIndex {
                base: g0_ptr,
                index: 0,
            },
            Span::UNDEFINED,
        );
        let g0_val = func
            .expressions
            .append(Expression::Load { pointer: g0_field }, Span::UNDEFINED);

        let pc_ptr = func
            .expressions
            .append(Expression::GlobalVariable(pc_var), Span::UNDEFINED);
        let pc_field = func.expressions.append(
            Expression::AccessIndex {
                base: pc_ptr,
                index: 0,
            },
            Span::UNDEFINED,
        );
        let pc_val = func
            .expressions
            .append(Expression::Load { pointer: pc_field }, Span::UNDEFINED);

        let sum = func.expressions.append(
            Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: g0_val,
                right: pc_val,
            },
            Span::UNDEFINED,
        );

        let out_ptr = func
            .expressions
            .append(Expression::GlobalVariable(high_group_var), Span::UNDEFINED);
        let out_field = func.expressions.append(
            Expression::AccessIndex {
                base: out_ptr,
                index: 0,
            },
            Span::UNDEFINED,
        );

        let mut body = Block::new();
        body.push(
            Statement::Emit(naga::Range::new_from_bounds(g0_val, g0_val)),
            Span::UNDEFINED,
        );
        body.push(
            Statement::Emit(naga::Range::new_from_bounds(pc_val, sum)),
            Span::UNDEFINED,
        );
        body.push(
            Statement::Store {
                pointer: out_field,
                value: sum,
            },
            Span::UNDEFINED,
        );
        func.body = body;

        module.entry_points.push(EntryPoint {
            name: "main".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        module
    }

    fn validate_wgsl_roundtrip(wgsl_src: &str) {
        let reparsed = wgsl_in::parse_str(wgsl_src)
            .unwrap_or_else(|err| panic!("emitted WGSL failed to re-parse: {err}\n{wgsl_src}"));
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        validator.validate(&reparsed).unwrap_or_else(|err| {
            panic!(
                "re-parsed WGSL failed baseline WebGPU validation:\n{}\nWGSL:\n{wgsl_src}",
                err.emit_to_string(wgsl_src)
            )
        });
    }

    #[test]
    fn lowers_push_constants_and_high_bind_groups_to_valid_baseline_wgsl() {
        let mut module = build_synthetic_module_with_push_constants_and_high_group();

        // Confirm unlowered module fails baseline Capabilities::empty() validation due to Immediate.
        let mut pre_validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        assert!(
            pre_validator.validate(&module).is_err(),
            "unlowered module with AddressSpace::Immediate must fail baseline WebGPU validation"
        );

        let report = lower_module_for_webgpu(&mut module);
        assert_eq!(report.total_remaps(), 2);

        // Push constant lowered into @group(3) @binding(0).
        assert_eq!(
            report.push_constant_remaps,
            vec![PushConstantRemap {
                variable_name: Some("push_data".to_string()),
                new_group: 3,
                new_binding: 0,
            }]
        );

        // Group 5 @binding(2) compressed into the least-populated group in 0..=3 (group 1 @binding(0),
        // because group 0 has 1 binding and group 3 now has 1 binding from Pass 1).
        assert_eq!(
            report.bind_group_remaps,
            vec![BindGroupRemap {
                variable_name: Some("high_group_out".to_string()),
                old_group: 5,
                old_binding: 2,
                new_group: 1,
                new_binding: 0,
            }]
        );

        // No AddressSpace::Immediate or group >= 4 remains.
        for (_, var) in module.global_variables.iter() {
            assert_ne!(var.space, AddressSpace::Immediate);
            if let Some(binding) = var.binding {
                assert!(binding.group < WEBGPU_MAX_BIND_GROUPS);
            }
        }

        // Validate under strict baseline WebGPU capabilities (Capabilities::empty()).
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        let info = validator
            .validate(&module)
            .expect("lowered module must pass baseline WebGPU validation");

        let wgsl_src = wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())
            .expect("lowered module must emit valid WGSL");

        assert!(
            wgsl_src.contains("@group(3) @binding(0)"),
            "expected @group(3) @binding(0) in WGSL:\n{wgsl_src}"
        );
        assert!(
            wgsl_src.contains("@group(1) @binding(0)"),
            "expected remapped @group(1) @binding(0) in WGSL:\n{wgsl_src}"
        );
        assert!(
            !wgsl_src.contains("var<immediate>"),
            "lowered WGSL must not contain var<immediate>:\n{wgsl_src}"
        );
        assert!(
            !wgsl_src.contains("@group(5)"),
            "lowered WGSL must not contain @group(5):\n{wgsl_src}"
        );
        validate_wgsl_roundtrip(&wgsl_src);
    }

    #[test]
    fn push_constant_avoids_existing_group_3_bindings_and_multiple_high_groups_balance() {
        let mut module = Module::default();
        let u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );

        // Pre-populate group 3 with bindings 0 and 4.
        for b in [0u32, 4u32] {
            module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("g3_b{b}")),
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding {
                        group: 3,
                        binding: b,
                    }),
                    ty: u32_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
        }

        // Add push constant -> should get group 3, binding 5 (max_existing + 1).
        module.global_variables.append(
            GlobalVariable {
                name: Some("pc".to_string()),
                space: AddressSpace::Immediate,
                binding: None,
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        // Add bindings in groups 4, 6, 7 -> should deterministically distribute across 0, 1, 2.
        for (g, b) in [(6u32, 1u32), (4u32, 0u32), (7u32, 3u32)] {
            module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("hi_g{g}_b{b}")),
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding {
                        group: g,
                        binding: b,
                    }),
                    ty: u32_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
        }

        let report = lower_module_for_webgpu(&mut module);
        assert_eq!(
            report.push_constant_remaps,
            vec![PushConstantRemap {
                variable_name: Some("pc".to_string()),
                new_group: 3,
                new_binding: 5,
            }]
        );

        // Sorted order of overflow bindings: (4,0), (6,1), (7,3) -> assigned to groups 0, 1, 2.
        assert_eq!(
            report.bind_group_remaps,
            vec![
                BindGroupRemap {
                    variable_name: Some("hi_g4_b0".to_string()),
                    old_group: 4,
                    old_binding: 0,
                    new_group: 0,
                    new_binding: 0,
                },
                BindGroupRemap {
                    variable_name: Some("hi_g6_b1".to_string()),
                    old_group: 6,
                    old_binding: 1,
                    new_group: 1,
                    new_binding: 0,
                },
                BindGroupRemap {
                    variable_name: Some("hi_g7_b3".to_string()),
                    old_group: 7,
                    old_binding: 3,
                    new_group: 2,
                    new_binding: 0,
                },
            ]
        );
    }

    #[test]
    fn push_constant_falls_back_to_lowest_free_group_when_group_3_is_full() {
        let mut module = Module::default();
        let u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );

        // Fill group 3 to WEBGPU_MAX_BINDINGS_PER_GROUP.
        for b in 0..(WEBGPU_MAX_BINDINGS_PER_GROUP as u32) {
            module.global_variables.append(
                GlobalVariable {
                    name: None,
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding {
                        group: 3,
                        binding: b,
                    }),
                    ty: u32_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
        }
        // Put one binding in group 0 so next free binding in group 0 is 1.
        module.global_variables.append(
            GlobalVariable {
                name: None,
                space: AddressSpace::Uniform,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        module.global_variables.append(
            GlobalVariable {
                name: Some("pc_fallback".to_string()),
                space: AddressSpace::Immediate,
                binding: None,
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        let report = lower_module_for_webgpu(&mut module);
        assert_eq!(
            report.push_constant_remaps,
            vec![PushConstantRemap {
                variable_name: Some("pc_fallback".to_string()),
                new_group: 0,
                new_binding: 1,
            }]
        );
    }

    #[test]
    fn complex_3d_pbr_fragment_shader_with_derivatives_textures_storage_and_high_groups_roundtrips() {
        let src = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
}

struct LightEntry {
    pos_radius: vec4<f32>,
    color_intensity: vec4<f32>,
}

struct LightBuffer {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    lights: array<LightEntry>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(4) @binding(0) var albedo_tex: texture_2d<f32>;
@group(5) @binding(1) var linear_sampler: sampler;
@group(6) @binding(0) var<storage, read> light_buf: LightBuffer;

fn compute_normal_scale(uv: vec2<f32>) -> f32 {
    let dx = dpdx(uv);
    let dy = dpdy(uv);
    return length(dx) + length(dy) + fwidth(uv.x);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>, @location(1) world_pos: vec3<f32>) -> @location(0) vec4<f32> {
    let base_col = textureSample(albedo_tex, linear_sampler, uv);
    let mip_col = textureSampleBias(albedo_tex, linear_sampler, uv, 0.25);
    let deriv = compute_normal_scale(uv);
    var accum = (base_col.rgb + mip_col.rgb) * 0.5;
    if (light_buf.count > 0u) {
        let l0 = light_buf.lights[0];
        let to_light = normalize(l0.pos_radius.xyz - world_pos);
        let ndl = max(dot(to_light, camera.sun_dir.xyz), 0.0);
        accum = accum + l0.color_intensity.rgb * ndl;
    }
    return vec4<f32>(accum + vec3<f32>(deriv * 0.001), 1.0);
}
"#;
        let mut module = wgsl_in::parse_str(src).expect("valid synthetic 3D PBR fragment WGSL");
        let (lowered_wgsl, report) =
            lower_and_emit_wgsl(&mut module).expect("3D PBR fragment shader must lower cleanly");

        assert_eq!(report.bind_group_remaps.len(), 3);
        assert!(report.push_constant_remaps.is_empty());
        assert!(!lowered_wgsl.contains("@group(4)"));
        assert!(!lowered_wgsl.contains("@group(5)"));
        assert!(!lowered_wgsl.contains("@group(6)"));
        validate_wgsl_roundtrip(&lowered_wgsl);

        // Re-running lowering on a fresh parse produces byte-identical WGSL.
        let mut module2 = wgsl_in::parse_str(src).expect("second parse");
        let (lowered_wgsl_2, report2) =
            lower_and_emit_wgsl(&mut module2).expect("second lowering must succeed");
        assert_eq!(report, report2);
        assert_eq!(lowered_wgsl, lowered_wgsl_2);
    }

    #[test]
    fn complex_3d_compute_shader_with_atomics_storage_texture_workgroup_and_high_groups_roundtrips()
    {
        let src = r#"
struct CounterBuffer {
    total: atomic<u32>,
    peak: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> counters: CounterBuffer;
@group(5) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(7) @binding(2) var depth_tex: texture_2d<f32>;
@group(7) @binding(3) var point_sampler: sampler;

var<workgroup> tile_accum: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    if (lidx == 0u) {
        atomicStore(&tile_accum, 0u);
    }
    workgroupBarrier();

    let uv = vec2<f32>(f32(gid.x) * 0.01, f32(gid.y) * 0.01);
    let sampled = textureSampleLevel(depth_tex, point_sampler, uv, 0.0);
    let gathered = textureGather(0, depth_tex, point_sampler, uv);
    let score = u32(max(sampled.x + gathered.x, 0.0) * 100.0);
    atomicAdd(&tile_accum, score);
    workgroupBarrier();

    if (lidx == 0u) {
        let sum = atomicLoad(&tile_accum);
        let prev = atomicAdd(&counters.total, sum);
        let _cmp = atomicCompareExchangeWeak(&counters.peak, prev, prev + sum);
    }
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), sampled);
}
"#;
        let mut module = wgsl_in::parse_str(src).expect("valid synthetic 3D compute WGSL");
        let (lowered_wgsl, report) =
            lower_and_emit_wgsl(&mut module).expect("3D compute shader must lower cleanly");

        assert_eq!(report.bind_group_remaps.len(), 3);
        assert!(!lowered_wgsl.contains("@group(5)"));
        assert!(!lowered_wgsl.contains("@group(7)"));
        validate_wgsl_roundtrip(&lowered_wgsl);
    }

    #[test]
    fn lowering_is_deterministic_and_byte_identical_across_global_insertion_orders() {
        let build_with_order = |reverse_globals: bool| -> (String, LoweringReport) {
            let mut module = Module::default();
            let vec4_ty = module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size: VectorSize::Quad,
                        scalar: Scalar::F32,
                    },
                },
                Span::UNDEFINED,
            );
            let block_ty = module.types.insert(
                Type {
                    name: Some("Params".to_string()),
                    inner: TypeInner::Struct {
                        members: vec![StructMember {
                            name: Some("v".to_string()),
                            ty: vec4_ty,
                            binding: None,
                            offset: 0,
                        }],
                        span: 16,
                    },
                },
                Span::UNDEFINED,
            );

            let mut specs = vec![
                ("pc_alpha", AddressSpace::Immediate, None),
                ("pc_beta", AddressSpace::Immediate, None),
                (
                    "hi_a",
                    AddressSpace::Uniform,
                    Some(ResourceBinding {
                        group: 4,
                        binding: 1,
                    }),
                ),
                (
                    "hi_b",
                    AddressSpace::Uniform,
                    Some(ResourceBinding {
                        group: 6,
                        binding: 0,
                    }),
                ),
                (
                    "hi_c",
                    AddressSpace::Uniform,
                    Some(ResourceBinding {
                        group: 5,
                        binding: 2,
                    }),
                ),
            ];
            if reverse_globals {
                specs.reverse();
            }

            let mut handles_by_name = BTreeMap::new();
            for (name, space, binding) in specs {
                let h = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.to_string()),
                        space,
                        binding,
                        ty: block_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                handles_by_name.insert(name, h);
            }

            let out_var = module.global_variables.append(
                GlobalVariable {
                    name: Some("out_buf".to_string()),
                    space: AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE,
                    },
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 0,
                    }),
                    ty: block_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );

            let mut func = Function {
                name: Some("main".to_string()),
                ..Function::default()
            };
            let mut body = Block::new();

            // Accumulate all 5 inputs in canonical name order.
            let mut acc: Option<Handle<Expression>> = None;
            for name in ["hi_a", "hi_b", "hi_c", "pc_alpha", "pc_beta"] {
                let g = handles_by_name[name];
                let ptr = func
                    .expressions
                    .append(Expression::GlobalVariable(g), Span::UNDEFINED);
                let field = func.expressions.append(
                    Expression::AccessIndex {
                        base: ptr,
                        index: 0,
                    },
                    Span::UNDEFINED,
                );
                let val = func
                    .expressions
                    .append(Expression::Load { pointer: field }, Span::UNDEFINED);
                body.push(
                    Statement::Emit(naga::Range::new_from_bounds(val, val)),
                    Span::UNDEFINED,
                );
                acc = Some(match acc {
                    None => val,
                    Some(prev) => {
                        let sum = func.expressions.append(
                            Expression::Binary {
                                op: naga::BinaryOperator::Add,
                                left: prev,
                                right: val,
                            },
                            Span::UNDEFINED,
                        );
                        body.push(
                            Statement::Emit(naga::Range::new_from_bounds(sum, sum)),
                            Span::UNDEFINED,
                        );
                        sum
                    }
                });
            }

            let out_ptr = func
                .expressions
                .append(Expression::GlobalVariable(out_var), Span::UNDEFINED);
            let out_field = func.expressions.append(
                Expression::AccessIndex {
                    base: out_ptr,
                    index: 0,
                },
                Span::UNDEFINED,
            );
            body.push(
                Statement::Store {
                    pointer: out_field,
                    value: acc.expect("accumulator"),
                },
                Span::UNDEFINED,
            );
            func.body = body;

            module.entry_points.push(EntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                early_depth_test: None,
                workgroup_size: [1, 1, 1],
                workgroup_size_overrides: None,
                function: func,
                mesh_info: None,
                task_payload: None,
                incoming_ray_payload: None,
            });

            lower_and_emit_wgsl(&mut module).expect("must lower deterministically")
        };

        let (wgsl_fwd, report_fwd) = build_with_order(false);
        let (wgsl_rev, report_rev) = build_with_order(true);

        assert_eq!(report_fwd, report_rev);
        assert_eq!(report_fwd.canonicalized(), report_fwd);
        validate_wgsl_roundtrip(&wgsl_fwd);
        validate_wgsl_roundtrip(&wgsl_rev);

        // Verify both runs mapped identical named bindings to identical slots.
        for name in ["pc_alpha", "pc_beta"] {
            let r1 = report_fwd
                .push_constant_remaps
                .iter()
                .find(|r| r.variable_name.as_deref() == Some(name))
                .expect("forward push constant remap");
            let r2 = report_rev
                .push_constant_remaps
                .iter()
                .find(|r| r.variable_name.as_deref() == Some(name))
                .expect("reverse push constant remap");
            assert_eq!(r1, r2);
        }
        // Repeated lowering on identical input is strictly byte-identical.
        let (wgsl_fwd_again, _) = build_with_order(false);
        assert_eq!(wgsl_fwd, wgsl_fwd_again);
    }

    #[test]
    fn remaps_colliding_immediate_pointer_type_without_panicking() {
        let mut module = build_synthetic_module_with_push_constants_and_high_group();
        let pc_struct_ty = module
            .types
            .iter()
            .find_map(|(h, t)| (t.name.as_deref() == Some("RootConstants")).then_some(h))
            .expect("RootConstants type");

        // Pre-insert BOTH a Uniform pointer AND an Immediate pointer to the exact same struct.
        let uniform_ptr_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Pointer {
                    base: pc_struct_ty,
                    space: AddressSpace::Uniform,
                },
            },
            Span::UNDEFINED,
        );
        let immediate_ptr_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Pointer {
                    base: pc_struct_ty,
                    space: AddressSpace::Immediate,
                },
            },
            Span::UNDEFINED,
        );

        // Part A: Helper function taking `ptr<immediate, RootConstants>` as an argument
        // must remap its argument type handle cleanly to `uniform_ptr_ty` without panicking
        // inside `UniqueArena::replace`.
        let mut module_with_helper = module.clone();
        let helper_fn = Function {
            name: Some("read_pc_helper".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("p".to_string()),
                ty: immediate_ptr_ty,
                binding: None,
            }],
            ..Function::default()
        };
        let helper_handle = module_with_helper
            .functions
            .append(helper_fn, Span::UNDEFINED);

        let report_a = try_lower_module_for_webgpu(&mut module_with_helper)
            .expect("colliding immediate pointer type must remap cleanly without panicking");
        assert_eq!(report_a.push_constant_remaps.len(), 1);
        assert_eq!(
            module_with_helper.functions[helper_handle].arguments[0].ty,
            uniform_ptr_ty
        );

        // Part B: Module with both colliding pointer types in `module.types` lowers and
        // round-trips cleanly through baseline WGSL validation.
        let (wgsl_src, report_b) = lower_and_emit_wgsl(&mut module)
            .expect("module with colliding immediate/uniform pointer types must emit valid WGSL");
        assert_eq!(report_b.push_constant_remaps.len(), 1);
        validate_wgsl_roundtrip(&wgsl_src);
    }

    #[test]
    fn complex_3d_skinned_vertex_shader_with_texture_sample_level_and_high_groups_roundtrips() {
        let src = r#"
struct FrameUniform {
    view_proj: mat4x4<f32>,
}

struct BonePalette {
    mats: array<mat4x4<f32>>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniform;
@group(4) @binding(0) var<storage, read> bones: BonePalette;
@group(6) @binding(1) var disp_tex: texture_2d<f32>;
@group(6) @binding(2) var disp_sampler: sampler;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) bone_idx: u32,
) -> VsOut {
    let skin_mat = bones.mats[bone_idx];
    let disp = textureSampleLevel(disp_tex, disp_sampler, uv, 0.0).x;
    let gathered = textureGather(0, disp_tex, disp_sampler, uv).x;
    let world_pos = skin_mat * vec4<f32>(pos + vec3<f32>(0.0, (disp + gathered) * 0.05, 0.0), 1.0);
    var out: VsOut;
    out.clip_pos = frame.view_proj * world_pos;
    out.uv = uv;
    return out;
}
"#;
        let mut m1 = wgsl_in::parse_str(src).expect("valid synthetic 3D vertex WGSL");
        let (wgsl1, report1) =
            lower_and_emit_wgsl(&mut m1).expect("3D skinned vertex shader must lower cleanly");
        assert_eq!(report1.bind_group_remaps.len(), 3);
        assert!(!wgsl1.contains("@group(4)"));
        assert!(!wgsl1.contains("@group(6)"));
        validate_wgsl_roundtrip(&wgsl1);

        let mut m2 = wgsl_in::parse_str(src).expect("second parse");
        let (wgsl2, report2) =
            lower_and_emit_wgsl(&mut m2).expect("second vertex lowering must succeed");
        assert_eq!(report1, report2);
        assert_eq!(wgsl1, wgsl2);
    }

    #[test]
    fn empty_module_and_empty_emit_ranges_lower_without_panicking() {
        // 1. Completely empty module (zero globals, zero entry points).
        let mut empty_module = Module::default();
        let (empty_wgsl, report) =
            lower_and_emit_wgsl(&mut empty_module).expect("empty module must lower cleanly");
        assert!(report.is_empty());
        validate_wgsl_roundtrip(&empty_wgsl);

        // 2. Entry point with an empty Statement::Emit range and zero globals.
        let mut module = Module::default();
        let mut func = Function {
            name: Some("noop".to_string()),
            ..Function::default()
        };
        let empty_range = naga::Range::from_index_range(0..0, &func.expressions);
        func.body
            .push(Statement::Emit(empty_range), Span::UNDEFINED);
        module.entry_points.push(EntryPoint {
            name: "noop".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let (wgsl_src, report) =
            lower_and_emit_wgsl(&mut module).expect("empty Emit range must not panic");
        assert!(report.is_empty());
        validate_wgsl_roundtrip(&wgsl_src);
    }

    #[test]
    fn guards_out_of_bounds_emit_range_and_dangling_handles_without_panicking() {
        // Construct a Handle<Expression> from a separate arena so it is out of bounds in `func`.
        let mut foreign_arena: Arena<Expression> = Arena::new();
        let _e0 = foreign_arena.append(
            Expression::Literal(naga::Literal::U32(0)),
            Span::UNDEFINED,
        );
        let oob_expr = foreign_arena.append(
            Expression::Literal(naga::Literal::U32(1)),
            Span::UNDEFINED,
        );

        let mut module = Module::default();
        let mut func = Function {
            name: Some("bad_emit".to_string()),
            ..Function::default()
        };
        func.body.push(
            Statement::Emit(naga::Range::new_from_bounds(oob_expr, oob_expr)),
            Span::UNDEFINED,
        );
        module.entry_points.push(EntryPoint {
            name: "bad_emit".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let err = try_lower_module_for_webgpu(&mut module).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidExpressionRange { .. }),
            "expected InvalidExpressionRange, got {err:?}"
        );

        // Dangling GlobalVariable handle inside an expression.
        let mut foreign_globals: Arena<GlobalVariable> = Arena::new();
        let mut dummy_types: naga::UniqueArena<Type> = naga::UniqueArena::new();
        let dummy_ty = dummy_types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        let dangling_global = foreign_globals.append(
            GlobalVariable {
                name: Some("ghost".to_string()),
                space: AddressSpace::Uniform,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: dummy_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );

        let mut module2 = Module::default();
        let mut func2 = Function {
            name: Some("dangling".to_string()),
            ..Function::default()
        };
        func2
            .expressions
            .append(Expression::GlobalVariable(dangling_global), Span::UNDEFINED);
        module2.entry_points.push(EntryPoint {
            name: "dangling".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: func2,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let err2 = try_lower_module_for_webgpu(&mut module2).unwrap_err();
        assert!(
            matches!(
                err2,
                LoweringError::InvalidHandle {
                    arena: "global_variables",
                    ..
                }
            ),
            "expected InvalidHandle for global_variables, got {err2:?}"
        );
    }

    #[test]
    fn guards_zero_sized_structs_zero_stride_arrays_and_size_overflow() {
        // 1. Zero-sized empty struct.
        let mut m_empty_struct = Module::default();
        m_empty_struct.types.insert(
            Type {
                name: Some("EmptyStruct".to_string()),
                inner: TypeInner::Struct {
                    members: Vec::new(),
                    span: 0,
                },
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_empty_struct).unwrap_err();
        assert!(
            matches!(err, LoweringError::ZeroSizedOrInvalidType { .. }),
            "expected ZeroSizedOrInvalidType for empty struct, got {err:?}"
        );

        // 2. Zero-stride array.
        let mut m_zero_stride = Module::default();
        let u32_ty = m_zero_stride.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        m_zero_stride.types.insert(
            Type {
                name: Some("ZeroStrideArr".to_string()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(4).expect("non-zero")),
                    stride: 0,
                },
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_zero_stride).unwrap_err();
        assert!(
            matches!(err, LoweringError::ZeroSizedOrInvalidType { .. }),
            "expected ZeroSizedOrInvalidType for zero-stride array, got {err:?}"
        );

        // 3. Array byte size overflow (`stride * count > u32::MAX`).
        let mut m_overflow = Module::default();
        let u32_ty = m_overflow.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        m_overflow.types.insert(
            Type {
                name: Some("HugeArr".to_string()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(1024).expect("non-zero")),
                    stride: u32::MAX,
                },
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_overflow).unwrap_err();
        assert!(
            matches!(err, LoweringError::ZeroSizedOrInvalidType { .. }),
            "expected ZeroSizedOrInvalidType for overflowing array, got {err:?}"
        );
    }

    #[test]
    fn guards_missing_and_duplicate_resource_bindings_and_u32_max_slot_wrap() {
        // 1. Missing @group/@binding on a uniform global.
        let mut m_missing = Module::default();
        let u32_ty = m_missing.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        m_missing.global_variables.append(
            GlobalVariable {
                name: Some("unbound_cbv".to_string()),
                space: AddressSpace::Uniform,
                binding: None,
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_missing).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidResourceBinding { .. }),
            "expected InvalidResourceBinding, got {err:?}"
        );

        // 2. Duplicate (group, binding) collision.
        let mut m_dup = Module::default();
        let u32_ty = m_dup.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        for name in ["first_cbv", "second_cbv"] {
            m_dup.global_variables.append(
                GlobalVariable {
                    name: Some(name.to_string()),
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 2,
                    }),
                    ty: u32_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
        }
        let err = try_lower_module_for_webgpu(&mut m_dup).unwrap_err();
        assert!(
            matches!(
                err,
                LoweringError::DuplicateResourceBinding {
                    group: 0,
                    binding: 2,
                    ..
                }
            ),
            "expected DuplicateResourceBinding, got {err:?}"
        );

        // 3. `u32::MAX` already occupied in group 3 -> safely scans to slot 0 without wrapping loop.
        let mut m_max = Module::default();
        let u32_ty = m_max.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        m_max.global_variables.append(
            GlobalVariable {
                name: Some("max_slot".to_string()),
                space: AddressSpace::Uniform,
                binding: Some(ResourceBinding {
                    group: 3,
                    binding: u32::MAX,
                }),
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        m_max.global_variables.append(
            GlobalVariable {
                name: Some("pc_wrap_safe".to_string()),
                space: AddressSpace::Immediate,
                binding: None,
                ty: u32_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let report = try_lower_module_for_webgpu(&mut m_max)
            .expect("u32::MAX occupied must safely allocate slot 0");
        assert_eq!(
            report.push_constant_remaps,
            vec![PushConstantRemap {
                variable_name: Some("pc_wrap_safe".to_string()),
                new_group: 3,
                new_binding: 0,
            }]
        );
    }

    #[test]
    fn guards_invalid_storage_texture_atomic_and_derivative_ops() {
        // 1. Implicit-derivative `textureSample` inside a helper called from Compute.
        let mut m_tex = Module::default();
        let vec2_ty = m_tex.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: Scalar::F32,
                },
            },
            Span::UNDEFINED,
        );
        let img_ty = m_tex.types.insert(
            Type {
                name: None,
                inner: TypeInner::Image {
                    dim: ImageDimension::D2,
                    arrayed: false,
                    class: ImageClass::Sampled {
                        kind: ScalarKind::Float,
                        multi: false,
                    },
                },
            },
            Span::UNDEFINED,
        );
        let samp_ty = m_tex.types.insert(
            Type {
                name: None,
                inner: TypeInner::Sampler { comparison: false },
            },
            Span::UNDEFINED,
        );
        let tex_var = m_tex.global_variables.append(
            GlobalVariable {
                name: Some("tex".to_string()),
                space: AddressSpace::Handle,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: img_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let samp_var = m_tex.global_variables.append(
            GlobalVariable {
                name: Some("samp".to_string()),
                space: AddressSpace::Handle,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 1,
                }),
                ty: samp_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let mut helper = Function {
            name: Some("sample_helper".to_string()),
            ..Function::default()
        };
        let img_e = helper
            .expressions
            .append(Expression::GlobalVariable(tex_var), Span::UNDEFINED);
        let samp_e = helper
            .expressions
            .append(Expression::GlobalVariable(samp_var), Span::UNDEFINED);
        let uv_e = helper
            .expressions
            .append(Expression::ZeroValue(vec2_ty), Span::UNDEFINED);
        let _sample_e = helper.expressions.append(
            Expression::ImageSample {
                image: img_e,
                sampler: samp_e,
                gather: None,
                coordinate: uv_e,
                array_index: None,
                offset: None,
                level: SampleLevel::Auto,
                depth_ref: None,
                clamp_to_edge: false,
            },
            Span::UNDEFINED,
        );
        let helper_h = m_tex.functions.append(helper, Span::UNDEFINED);

        let mut cs_fn = Function {
            name: Some("cs_main".to_string()),
            ..Function::default()
        };
        cs_fn.body.push(
            Statement::Call {
                function: helper_h,
                arguments: Vec::new(),
                result: None,
            },
            Span::UNDEFINED,
        );
        m_tex.entry_points.push(EntryPoint {
            name: "cs_main".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: cs_fn,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        let err = try_lower_module_for_webgpu(&mut m_tex).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidTextureOp { .. }),
            "expected InvalidTextureOp for compute textureSample(Auto), got {err:?}"
        );

        // 2. Derivative op (`dpdx`) in Compute stage.
        let mut m_deriv = Module::default();
        let f32_ty = m_deriv.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::UNDEFINED,
        );
        let mut cs_deriv = Function {
            name: Some("cs_deriv".to_string()),
            ..Function::default()
        };
        let z = cs_deriv
            .expressions
            .append(Expression::ZeroValue(f32_ty), Span::UNDEFINED);
        let _d = cs_deriv.expressions.append(
            Expression::Derivative {
                axis: DerivativeAxis::X,
                ctrl: DerivativeControl::None,
                expr: z,
            },
            Span::UNDEFINED,
        );
        m_deriv.entry_points.push(EntryPoint {
            name: "cs_deriv".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: cs_deriv,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        let err = try_lower_module_for_webgpu(&mut m_deriv).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidDerivativeOp { .. }),
            "expected InvalidDerivativeOp for compute dpdx, got {err:?}"
        );

        // 3. Atomic in read-only storage buffer.
        let mut m_ro_atomic = Module::default();
        let atomic_u32 = m_ro_atomic.types.insert(
            Type {
                name: None,
                inner: TypeInner::Atomic(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        m_ro_atomic.global_variables.append(
            GlobalVariable {
                name: Some("ro_atomic".to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: atomic_u32,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_ro_atomic).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidAtomicOp { .. }),
            "expected InvalidAtomicOp for read-only storage atomic, got {err:?}"
        );

        // 4. Push constant holding an atomic -> UnsupportedPushConstantType + rollback check.
        let mut m_pc_atomic = Module::default();
        let atomic_u32 = m_pc_atomic.types.insert(
            Type {
                name: None,
                inner: TypeInner::Atomic(Scalar::U32),
            },
            Span::UNDEFINED,
        );
        let pc_h = m_pc_atomic.global_variables.append(
            GlobalVariable {
                name: Some("pc_atomic".to_string()),
                space: AddressSpace::Immediate,
                binding: None,
                ty: atomic_u32,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let snapshot = m_pc_atomic.clone();
        let err = try_lower_module_for_webgpu(&mut m_pc_atomic).unwrap_err();
        assert!(
            matches!(err, LoweringError::UnsupportedPushConstantType { .. }),
            "expected UnsupportedPushConstantType, got {err:?}"
        );
        // Verify transactional rollback preserved the original Immediate variable untouched.
        assert_eq!(m_pc_atomic.global_variables[pc_h], snapshot.global_variables[pc_h]);

        // 5. Store into read-only storage buffer -> InvalidStorageOp.
        let mut m_ro_store = Module::default();
        let f32_ty2 = m_ro_store.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::UNDEFINED,
        );
        let ro_var = m_ro_store.global_variables.append(
            GlobalVariable {
                name: Some("ro_buf".to_string()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: f32_ty2,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let mut f_store = Function {
            name: Some("bad_store".to_string()),
            ..Function::default()
        };
        let ptr = f_store
            .expressions
            .append(Expression::GlobalVariable(ro_var), Span::UNDEFINED);
        let zero = f_store
            .expressions
            .append(Expression::ZeroValue(f32_ty2), Span::UNDEFINED);
        f_store
            .body
            .push(Statement::Store { pointer: ptr, value: zero }, Span::UNDEFINED);
        m_ro_store.entry_points.push(EntryPoint {
            name: "bad_store".to_string(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function: f_store,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        let err = try_lower_module_for_webgpu(&mut m_ro_store).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidStorageOp { .. }),
            "expected InvalidStorageOp for store to read-only storage, got {err:?}"
        );
    }

    #[test]
    fn guards_unsupported_cube_storage_texture_and_atomic_image_ops() {
        // Cube storage texture is forbidden in WebGPU.
        let mut m_cube = Module::default();
        let cube_storage_ty = m_cube.types.insert(
            Type {
                name: None,
                inner: TypeInner::Image {
                    dim: ImageDimension::Cube,
                    arrayed: false,
                    class: ImageClass::Storage {
                        format: StorageFormat::Rgba8Unorm,
                        access: StorageAccess::STORE,
                    },
                },
            },
            Span::UNDEFINED,
        );
        m_cube.global_variables.append(
            GlobalVariable {
                name: Some("cube_uav".to_string()),
                space: AddressSpace::Handle,
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: cube_storage_ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_cube).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidTextureOp { .. }),
            "expected InvalidTextureOp for cube storage texture, got {err:?}"
        );

        // 64-bit atomic scalar type is forbidden in baseline WebGPU.
        let mut m_a64 = Module::default();
        m_a64.types.insert(
            Type {
                name: Some("AtomicU64".to_string()),
                inner: TypeInner::Atomic(Scalar::U64),
            },
            Span::UNDEFINED,
        );
        let err = try_lower_module_for_webgpu(&mut m_a64).unwrap_err();
        assert!(
            matches!(err, LoweringError::InvalidAtomicOp { .. }),
            "expected InvalidAtomicOp for 64-bit atomic, got {err:?}"
        );
    }
}
