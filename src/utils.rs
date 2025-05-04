use crate::solana_export::pubkey::Pubkey;

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
/// use fast_instruction::solana_export::pubkey::Pubkey;
/// use fast_instruction::utils::fast_cmp_pubkey;
///
/// let pubkey1 = Pubkey::new_unique();
/// let pubkey2 = pubkey1;
/// assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
/// ```
#[inline(always)]
pub fn fast_cmp_pubkey(a: &Pubkey, b: &Pubkey) -> bool {
    // we could save a few CUs (~3) doing this in assembly as the compiler
    // tries to be cute and optimize for the case where we can exit early
    // at the cost of the full comparison. Not worth it really though
    unsafe {
        let a: &[u64; 4] = std::mem::transmute(a);
        let b: &[u64; 4] = std::mem::transmute(b);

        a == b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[quickcheck_macros::quickcheck]
    #[allow(clippy::missing_transmute_annotations)]
    fn quickcheck_fast_cmp_pubkey_equal(a: (u64, u64, u64, u64), b: (u64, u64, u64, u64)) {
        let a = unsafe { std::mem::transmute(a) };
        let b = unsafe { std::mem::transmute(b) };
        let pubkey_a = Pubkey::new_from_array(a);
        let pubkey_b = Pubkey::new_from_array(b);

        assert_eq!(a == b, pubkey_a == pubkey_b);
        assert_eq!(fast_cmp_pubkey(&pubkey_a, &pubkey_b), pubkey_a == pubkey_b);
    }

    #[test]
    fn test_fast_cmp_pubkey_equal() {
        let pubkey1 = Pubkey::new_unique();
        let pubkey2 = pubkey1;
        assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
    }

    #[test]
    fn test_fast_cmp_pubkey_not_equal() {
        let pubkey1 = Pubkey::new_unique();
        let pubkey2 = Pubkey::new_unique();
        assert!(!fast_cmp_pubkey(&pubkey1, &pubkey2));
    }

    #[test]
    fn test_fast_cmp_pubkey_zero() {
        let zero_pubkey = Pubkey::default();
        assert!(fast_cmp_pubkey(&zero_pubkey, &zero_pubkey));
        assert!(!fast_cmp_pubkey(&zero_pubkey, &Pubkey::new_unique()));
    }

    #[test]
    fn test_fast_cmp_pubkey_known_values() {
        let pubkey1 = Pubkey::new_from_array([1; 32]);
        let pubkey2 = Pubkey::new_from_array([1; 32]);
        let pubkey3 = Pubkey::new_from_array([2; 32]);

        assert!(fast_cmp_pubkey(&pubkey1, &pubkey2));
        assert!(!fast_cmp_pubkey(&pubkey1, &pubkey3));
    }
}
