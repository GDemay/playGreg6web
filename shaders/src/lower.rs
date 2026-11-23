use std::collections::BTreeSet;

use naga::{AddressSpace, Handle, ResourceBinding, Type, TypeInner};

use crate::compat::WEBGPU_MAX_BIND_GROUPS;

/// Preferred WebGPU bind group for lowered push constants / immediate data.
pub const PUSH_CONSTANT_TARGET_GROUP: u32 = WEBGPU_MAX_BIND_GROUPS - 1;

/// WebGPU baseline default `maxBindingsPerBindGroup`.
pub const WEBGPU_MAX_BINDINGS_PER_GROUP: usize = 1000;

/// Record of an `AddressSpace::Immediate` global variable lowered into a WebGPU uniform buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantRemap {
    pub variable_name: Option<String>,
    pub new_group: u32,
    pub new_binding: u32,
}

/// Record of an out-of-range (`group >= 4`) resource binding compressed into groups `0..=3`.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

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
/// and WGSL emission.
pub fn lower_module_for_webgpu(module: &mut naga::Module) -> LoweringReport {
    let push_constant_remaps = lower_push_constants(module);
    let bind_group_remaps = lower_bind_groups(module);
    LoweringReport {
        push_constant_remaps,
        bind_group_remaps,
    }
}

/// Pass 1: Lower any `AddressSpace::Immediate` global variables (D3D12 root constants / Vulkan
/// push constants) to `AddressSpace::Uniform` bindings in WebGPU-valid bind groups `0..=3`.
pub fn lower_push_constants(module: &mut naga::Module) -> Vec<PushConstantRemap> {
    let mut used = collect_used_bindings_in_webgpu_groups(module);

    let immediate_handles: Vec<Handle<naga::GlobalVariable>> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            if var.space == AddressSpace::Immediate {
                Some(handle)
            } else {
                None
            }
        })
        .collect();

    if immediate_handles.is_empty() {
        return Vec::new();
    }

    let mut remaps = Vec::with_capacity(immediate_handles.len());
    for handle in immediate_handles {
        let group = select_push_constant_group(&used);
        let binding = allocate_next_binding(&mut used[group as usize]);

        let var = &mut module.global_variables[handle];
        var.space = AddressSpace::Uniform;
        var.binding = Some(ResourceBinding { group, binding });

        remaps.push(PushConstantRemap {
            variable_name: var.name.clone(),
            new_group: group,
            new_binding: binding,
        });
    }

    // Also rewrite any standalone `Pointer`/`ValuePointer` types with `AddressSpace::Immediate`
    // into `AddressSpace::Uniform` when doing so does not collide with an existing type entry.
    rewrite_immediate_pointer_types(module);

    remaps
}

/// Pass 2: Deterministically compress any global variable binding with `group >= 4` into free
/// binding slots inside WebGPU baseline groups `0..=3`.
pub fn lower_bind_groups(module: &mut naga::Module) -> Vec<BindGroupRemap> {
    let mut used = collect_used_bindings_in_webgpu_groups(module);

    // Collect and sort all bindings with group >= 4 deterministically by (old_group, old_binding, handle_index).
    let mut overflow_bindings: Vec<(u32, u32, usize, Handle<naga::GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            let binding = var.binding?;
            if binding.group >= WEBGPU_MAX_BIND_GROUPS {
                Some((binding.group, binding.binding, handle.index(), handle))
            } else {
                None
            }
        })
        .collect();

    if overflow_bindings.is_empty() {
        return Vec::new();
    }

    overflow_bindings.sort_unstable_by_key(|&(g, b, idx, _)| (g, b, idx));

    let mut remaps = Vec::with_capacity(overflow_bindings.len());
    for (old_group, old_binding, _, handle) in overflow_bindings {
        // Choose the least-populated group in 0..=3, tie-breaking by lowest group index.
        let new_group = (0..(WEBGPU_MAX_BIND_GROUPS as usize))
            .min_by_key(|&g| (used[g].len(), g))
            .unwrap_or(0) as u32;
        let new_binding = allocate_next_binding(&mut used[new_group as usize]);

        let var = &mut module.global_variables[handle];
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

    remaps
}

fn collect_used_bindings_in_webgpu_groups(
    module: &naga::Module,
) -> [BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize] {
    let mut used: [BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize] = Default::default();
    for (_, var) in module.global_variables.iter() {
        if let Some(binding) = var.binding
            && binding.group < WEBGPU_MAX_BIND_GROUPS
        {
            used[binding.group as usize].insert(binding.binding);
        }
    }
    used
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
    (0..(WEBGPU_MAX_BIND_GROUPS as usize))
        .min_by_key(|&g| (used[g].len(), g))
        .unwrap_or(preferred) as u32
}

fn allocate_next_binding(occupied: &mut BTreeSet<u32>) -> u32 {
    let candidate = match occupied.last().copied() {
        None => 0,
        Some(max_existing) => match max_existing.checked_add(1) {
            Some(next) => next,
            None => {
                // `u32::MAX` is already taken; scan for the lowest free slot.
                let mut slot = 0u32;
                while occupied.contains(&slot) {
                    slot = slot.wrapping_add(1);
                }
                slot
            }
        },
    };
    occupied.insert(candidate);
    candidate
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

    for (handle, replacement) in candidates {
        let already_exists = module.types.iter().any(|(_, existing)| *existing == replacement);
        if !already_exists {
            module.types.replace(handle, replacement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::back::wgsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::{
        Block, EntryPoint, Expression, Function, GlobalVariable, MemoryDecorations, Module, Scalar,
        ShaderStage, Span, Statement, StorageAccess, StructMember, VectorSize,
    };

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
}
