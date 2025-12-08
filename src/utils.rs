use crate::solana_export::pubkey::Pubkey;
use core::ptr::read_unaligned;

/// Performs a fast comparison of two Solana public keys.
///
/// This function is designed to optimize the comparison of Solana public keys,
/// particularly for the BPF (Berkley Packet Filter) target used in Solana smart contracts.
/// It addresses performance issues with the standard equality check, which LLVM
/// struggles to optimize effectively for BPF.
///
/// # Safety
///
/// This function uses `unsafe` Rust to transmute the `Pubkey` structs into arrays of `u64`.
/// While this is generally safe given the known structure of `Pubkey`, it relies on
/// the internal representation of `Pubkey` remaining stable.
///
/// # Performance
///
/// This implementation saves approximately 3 compute units compared to the standard
/// equality check. While further optimizations could be achieved using assembly,
/// the marginal gains are considered not worth the added complexity.
///
/// # Arguments
///
/// * `a` - A reference to the first `Pubkey` to compare
/// * `b` - A reference to the second `Pubkey` to compare
///
/// # Returns
///
/// `true` if the public keys are equal, `false` otherwise.
///
/// # Example
///
/// ```
/// use peephole::solana_export::pubkey::Pubkey;
/// use peephole::utils::fast_cmp_pubkey;
///
/// let pubkey1 = Pubkey::default();
/// let pubkey2 = pubkey1;
/// assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
/// ```

pub fn fast_cmp_pubkey(a: &Pubkey, b: &Pubkey) -> bool {
    // Without it, LLVM recognizes the 32-byte comparison pattern and "optimizes"
    // it into a memcmp call, which is catastrophic on BPF (external syscall).
    // Even explicit element-wise `a[i] == b[i] && ...` gets pattern-matched back
    // to memcmp. XOR-OR patterns get pessimized into spilling intermediates to
    // the stack. black_box hints cause similar stack spills.
    //
    // read_unaligned prevents LLVM from reasoning about the loads, forcing it to
    // emit the straightforward load-compare-branch sequence we actually want.
    unsafe {
        let a = a as *const Pubkey as *const u64;
        let b = b as *const Pubkey as *const u64;

        read_unaligned(a) == read_unaligned(b)
            && read_unaligned(a.add(1)) == read_unaligned(b.add(1))
            && read_unaligned(a.add(2)) == read_unaligned(b.add(2))
            && read_unaligned(a.add(3)) == read_unaligned(b.add(3))
    }
}

#[cfg(test)]
mod tests {
    use crate::solana_export::{pubkey_from_array, unique_pubkey};

    use super::*;

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::missing_transmute_annotations)]
    fn quickcheck_fast_cmp_pubkey_equal(a: (u64, u64, u64, u64), b: (u64, u64, u64, u64)) {
        let a = unsafe { std::mem::transmute(a) };
        let b = unsafe { std::mem::transmute(b) };
        let pubkey_a = pubkey_from_array(a);
        let pubkey_b = pubkey_from_array(b);

        assert_eq!(a == b, pubkey_a == pubkey_b);
        assert_eq!(fast_cmp_pubkey(&pubkey_a, &pubkey_b), pubkey_a == pubkey_b);
    }

    #[test]
    fn test_fast_cmp_pubkey_equal() {
        let pubkey1 = unique_pubkey();
        let pubkey2 = pubkey1;
        assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
    }

    #[test]
    fn test_fast_cmp_pubkey_not_equal() {
        let pubkey1 = unique_pubkey();
        let pubkey2 = unique_pubkey();
        assert!(!fast_cmp_pubkey(&pubkey1, &pubkey2));
    }

    #[test]
    fn test_fast_cmp_pubkey_zero() {
        let zero_pubkey = Pubkey::default();
        assert!(fast_cmp_pubkey(&zero_pubkey, &zero_pubkey));
        assert!(!fast_cmp_pubkey(&zero_pubkey, &unique_pubkey()));
    }

    #[test]
    fn test_fast_cmp_pubkey_known_values() {
        let pubkey1 = pubkey_from_array([1; 32]);
        let pubkey2 = pubkey_from_array([1; 32]);
        let pubkey3 = pubkey_from_array([2; 32]);

        assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
        assert!(!fast_cmp_pubkey(&pubkey1, &pubkey3));
    }
}
