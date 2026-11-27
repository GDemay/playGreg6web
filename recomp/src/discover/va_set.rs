//! Ordered VA set for [`super::Discovery`]'s leaders and worklist.
//!
//! A fresh round inserts ~4M leaders and pops as many worklist entries, and every decoded block
//! asks for the next leader after its start. As `BTreeSet`s these were a fifth of the decode time.
//! [`VaSet`] keeps the VAs of one dense range (the executable sections) in a hierarchical bitmap,
//! one bit per byte, so insert, remove and "first at or after" are a few word operations; VAs
//! outside the range go to a `BTreeSet`. It behaves exactly like a `BTreeSet<u64>`.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Default)]
pub(super) struct VaSet {
    /// First VA of the bitmap range.
    lo: u64,
    /// Number of VAs in the bitmap range.
    span: u64,
    /// `levels[0]` has one bit per VA of the range; bit `i` of `levels[k + 1]` is set when word
    /// `i` of `levels[k]` is non-zero. The last level is a single word.
    levels: Vec<Vec<u64>>,
    /// VAs outside the bitmap range.
    outside: BTreeSet<u64>,
}

impl VaSet {
    /// An empty set whose bitmap covers `[lo, hi)`.
    pub fn with_range(lo: u64, hi: u64) -> Self {
        let span = hi.saturating_sub(lo);
        let mut levels = Vec::new();
        if span > 0 {
            let mut bits = span;
            loop {
                let words = bits.div_ceil(64);
                levels.push(vec![0u64; usize::try_from(words).unwrap_or(usize::MAX)]);
                if words <= 1 {
                    break;
                }
                bits = words;
            }
        }
        Self {
            lo,
            span,
            levels,
            outside: BTreeSet::new(),
        }
    }

    fn index(&self, va: u64) -> Option<u64> {
        let i = va.wrapping_sub(self.lo);
        (va >= self.lo && i < self.span).then_some(i)
    }

    /// Inserts `va`; returns whether it was new.
    pub fn insert(&mut self, va: u64) -> bool {
        let Some(mut i) = self.index(va) else {
            return self.outside.insert(va);
        };
        for (level, words) in self.levels.iter_mut().enumerate() {
            let word = &mut words[(i >> 6) as usize];
            let bit = 1u64 << (i & 63);
            let was = *word;
            *word |= bit;
            if level == 0 && was & bit != 0 {
                return false;
            }
            if was != 0 {
                // The levels above already mark this word.
                break;
            }
            i >>= 6;
        }
        true
    }

    /// Removes the in-range index `i` (which must be present).
    fn remove_index(&mut self, mut i: u64) {
        for words in &mut self.levels {
            let word = &mut words[(i >> 6) as usize];
            *word &= !(1u64 << (i & 63));
            if *word != 0 {
                break;
            }
            i >>= 6;
        }
    }

    /// The first in-range index at or after `i`.
    fn first_index_from(&self, i: u64) -> Option<u64> {
        let mut pos = i;
        let mut level = 0;
        loop {
            let words = self.levels.get(level)?;
            let w = usize::try_from(pos >> 6).ok()?;
            let word = *words.get(w)? & (!0u64 << (pos & 63));
            if word != 0 {
                let mut found = ((w as u64) << 6) | u64::from(word.trailing_zeros());
                for lower in self.levels[..level].iter().rev() {
                    found = (found << 6) | u64::from(lower[found as usize].trailing_zeros());
                }
                return Some(found);
            }
            pos = w as u64 + 1;
            level += 1;
        }
    }

    /// The smallest element at or after `va` (`set.range(va..).next()`).
    pub fn first_from(&self, va: u64) -> Option<u64> {
        let hi = self.lo + self.span;
        if self.span == 0 || va >= hi {
            return self.outside.range(va..).next().copied();
        }
        if va < self.lo
            && let Some(&below) = self.outside.range(va..self.lo).next()
        {
            return Some(below);
        }
        let from = va.max(self.lo) - self.lo;
        if let Some(i) = self.first_index_from(from) {
            return Some(self.lo + i);
        }
        self.outside.range(hi..).next().copied()
    }

    /// Removes and returns the smallest element.
    pub fn pop_first(&mut self) -> Option<u64> {
        if let Some(&below) = self.outside.first()
            && (self.span == 0 || below < self.lo)
        {
            return self.outside.pop_first();
        }
        if let Some(i) = self.first_index_from(0) {
            self.remove_index(i);
            return Some(self.lo + i);
        }
        self.outside.pop_first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random operations against a `BTreeSet` reference.
    #[test]
    fn matches_btreeset() {
        for (lo, hi) in [
            (0x1000, 0x1000),
            (0x1000, 0x1001),
            (0x1000, 0x1040),
            (0x1000, 0x9_1234),
        ] {
            let mut set = VaSet::with_range(lo, hi);
            let mut reference = BTreeSet::new();
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            for _ in 0..200_000 {
                let r = next();
                let va = 0xff0 + r % (hi - lo + 0x40);
                match r >> 60 {
                    0..=7 => assert_eq!(set.insert(va), reference.insert(va)),
                    8..=11 => assert_eq!(set.first_from(va), reference.range(va..).next().copied()),
                    _ => assert_eq!(set.pop_first(), reference.pop_first()),
                }
            }
            while let Some(va) = reference.pop_first() {
                assert_eq!(set.pop_first(), Some(va));
            }
            assert_eq!(set.pop_first(), None);
            assert_eq!(set.first_from(0), None);
        }
    }
}
