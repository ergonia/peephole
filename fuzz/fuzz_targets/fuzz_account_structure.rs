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
    assert!(do_quickcheck_compare_with_solana_deserialize(
        data.account_types.clone(),
        data.instruction_data_gen.clone()
    ));
});
