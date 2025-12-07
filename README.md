# peephole

Zero-copy account parsing for Solana programs.

## Why?

`solana_program::entrypoint::deserialize` allocates `AccountInfo` structs and copies account data into them. `peephole` skips that—you get typed pointers directly into the runtime's buffer, so reads and writes happen in-place with no allocation overhead.

## Installation

```toml
[dependencies]
peephole = { version = "1.0", features = ["solana-sdk"] }
```

Feature flags: `solana-sdk` or `pinocchio-sdk` (exactly one required).

## Basic Iteration

The runtime passes your program a buffer containing all accounts followed by instruction data. `AccountIterator` walks this buffer, yielding each account:

```rust
use peephole::account_iterator::{AccountIterator, NextAccount, AccountInInstruction};

let mut iter = unsafe { AccountIterator::new_from_instruction(input) };

loop {
    match iter.next() {
        NextAccount::Account(AccountInInstruction::RealAccount(acc), next) => {
            // Full account with metadata and data
            let key = &acc.static_data.key;
            let owner = &acc.static_data.owner;
            let lamports = acc.static_data.lamports;
            let is_signer = acc.static_data.is_signer != 0;
            let data: &[u8] = acc.data();

            iter = next;
        }
        NextAccount::Account(AccountInInstruction::Dup(idx), next) => {
            // This account is a duplicate of account at index `idx`.
            // The runtime compresses duplicates to save space—only the
            // first occurrence has full data.
            iter = next;
        }
        NextAccount::Data(instruction_data) => {
            // All accounts consumed, this is your instruction data
            break;
        }
    }
}
```

## Typed Slurping

When you know an account's data layout, cast it directly to your struct:

```rust
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct TokenAccount {
    mint: [u8; 32],
    owner: [u8; 32],
    amount: u64,
}

let (account, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };

// Direct field access—no parsing, no copying
let amount = account.data.amount;
account.data.amount = new_amount;  // writes go directly to runtime buffer

// Metadata still available
let pubkey = &account.static_data.key;
let is_writable = account.static_data.is_writable != 0;
```

### Batch Slurping

Slurp N accounts of the same type as an array—single pointer cast for the whole batch:

```rust
let (accounts, iter) = unsafe { iter.static_slurp_typed_accounts::<TokenAccount, 3>() };

for acc in accounts.iter() {
    process(acc.data.amount);
}

// Or by index
let first = &accounts[0].data;
let second = &accounts[1].data;
```

### Requirements

- `T: Pod + Zeroable` (from `bytemuck`)
- `size_of::<T>() % 8 == 0` — accounts are 8-byte aligned in the buffer
- Each account's `data_len` must exactly equal `size_of::<T>()`
- All accounts in the range must be real (no duplicate markers)

## Safety Model

The unsafe methods assume you know the account layout. If you slurp a `TokenAccount` but the actual data is a `MintAccount`, you get a valid pointer to garbage—reads return wrong values, writes corrupt data silently.

The typical pattern: verify a trusted signer first, then assume account types:

```rust
let mut iter = unsafe { AccountIterator::new_from_instruction(input) };

// First account is always safe to read (can't be a dup)
let (authority, iter) = unsafe { iter.known_next_full_account() };

if !is_program_authority(&authority.static_data.key) {
    return Err(ProgramError::Unauthorized);
}

// Authority signed this transaction, so we trust the account layout
let (token, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
let (vault, iter) = unsafe { iter.static_slurp_typed_account::<VaultAccount>() };
```

Debug builds (`debug_assertions`) verify invariants: correct account counts, no unexpected duplicates, matching data sizes. Release builds skip these checks entirely.

## Buffer Layout

For reference, here's how the runtime serializes accounts:

```text
┌─────────────────────────────────────────────┐
│ num_accounts: u64                           │
├─────────────────────────────────────────────┤
│ Account 0                                   │
├─────────────────────────────────────────────┤
│ Account 1                                   │
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ instruction_data_len: u64                   │
├─────────────────────────────────────────────┤
│ instruction_data: [u8]                      │
└─────────────────────────────────────────────┘
```

Each account slot starts with a marker byte:
- `0xFF`: Real account with full metadata and data
- `0x00-0xFE`: Duplicate—the value is the index of the original

Real accounts are laid out as:

```text
┌──────────────────────────────────────────────┐
│ NonDupAccountStatic (128 bytes)              │
│   is_dup: u8          (always 0xFF)          │
│   is_signer: u8                              │
│   is_writable: u8                            │
│   executable: u8                             │
│   original_data_len: u32                     │
│   key: Pubkey                                │
│   owner: Pubkey                              │
│   lamports: u64                              │
│   data_len: u64                              │
├──────────────────────────────────────────────┤
│ data: [u8; data_len]                         │
├──────────────────────────────────────────────┤
│ _buffer: [u8; 10240]                         │
│   (reserved for realloc during execution)    │
├──────────────────────────────────────────────┤
│ padding to 8-byte alignment                  │
├──────────────────────────────────────────────┤
│ rent_epoch: u64                              │
└──────────────────────────────────────────────┘
```

The 10KB buffer after each account's data is `MAX_PERMITTED_DATA_INCREASE`—space reserved by the runtime for accounts that grow during execution.

## Types Reference

```rust
pub enum NextAccount {
    Account(AccountInInstruction, AccountIterator),
    Data(&'static mut [u8]),
}

pub enum AccountInInstruction {
    RealAccount(NonDupAccount<'static>),
    Dup(usize),
}

pub struct NonDupAccount<'a> {
    pub static_data: &'a NonDupAccountStatic,
    pub all_data: &'a mut [u8],
    pub rent_epoch: &'a u64,
}

pub struct TypedNonDupAccount<T: Pod + Zeroable> {
    pub static_data: NonDupAccountStatic,
    pub data: T,
    _buffer: [u8; MAX_PERMITTED_DATA_INCREASE],
    pub rent_epoch: u64,
}
```

## See Also

- **`pubkey_byte_map`**: O(1) pubkey lookup for up to 256 keys, indexed by first byte (must be unique)
- **`assume!`**: Macro that's `debug_assert!` in debug builds, `unreachable_unchecked` in release
