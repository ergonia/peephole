#![no_main]
use libfuzzer_sys::fuzz_target;

use fast_instruction::account_iterator::arbitrary_impls::{
    do_quickcheck_compare_with_solana_deserialize, do_quickcheck_mixed_account_types_with_arrays,
    TestAccountType,
};
#[derive(arbitrary::Arbitrary, Clone, Debug)]
pub struct AccountStructureInfo {
    pub account_types: Vec<TestAccountType>,
    pub instruction_data_gen: Vec<u8>,
}

fuzz_target!(|data: AccountStructureInfo| {
    assert!(do_quickcheck_mixed_account_types_with_arrays(
        data.account_types.clone(),
        data.instruction_data_gen.clone()
    ));
    // pinocchio parsing gets blown up by very trivial inputs, i suspect that
    // it makes assumptions about data layout/generation that are only true for svm
    // generated buffers? It repeatedly accesses one off the end of the buffer.
    // Adding padding doesn't do anything?
});
