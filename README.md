# peephole

Zero-copy account parsing for Solana programs.

## Why?

`solana_program::entrypoint::deserialize` allocates `AccountInfo` structs and copies account data. peephole skips that with a zero-alloc and zero-copy view over the account and instruction data.

Real-world: [51 CU oracle update](https://solscan.io/tx/JkGJc3Q2eAjPKWG4PjbNxYguqc6gAqD96bMaAmqrotH2dn5cWXV7w8Sgp7tbckr1QKqab6749rhgPjTnQEDwDkB), [67 CUs for 3 oracle updates](https://solscan.io/tx/4w5T3BrUUb2zmcuVwZjNaiHY2ysfMeMNa5NE9bHfoywc8iPTJCfcFndZ2C9TyGQTj3jLMaVRALbRDDpWV9HAEHhU),
[<20 CUs](https://solscan.io/tx/45Kzs3dkFQqq5BDB8qqp2F3QpBoeRLbzvnLBHZ6ZpL2EUJrBUzm4dcf4dquXt8dKvySHpHiPd5JMgKDmmKF3rn3r) for other production contracts.

## Installation

```toml
[dependencies]
peephole = { version = "1.0", features = ["solana-sdk"] }
```

Feature flags: `solana-sdk` or `pinocchio-sdk` (exactly one required).

## Basic Iteration

```rust
use peephole::account_iterator::{AccountIterator, NextAccount, AccountInInstruction};

let mut iter = unsafe { AccountIterator::new_from_instruction(input) };

loop {
    match iter.next() {
        NextAccount::Account(AccountInInstruction::RealAccount(acc), next) => {
            let key = &acc.static_data.key;
            let lamports = acc.static_data.lamports;
            let data: &[u8] = acc.data();
            iter = next;
        }
        NextAccount::Account(AccountInInstruction::Dup(idx), next) => {
            // Duplicate of account `idx`—runtime only serializes full data once
            iter = next;
        }
        NextAccount::Data(instruction_data, program_id) => {
            // instruction_data: &mut [u8], program_id: &Pubkey
            break;
        }
    }
}
```

## Typed Slurping

Cast account data directly to your struct:

```rust
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct TokenAccount {
    mint: [u8; 32],
    owner: [u8; 32],
    amount: u64,
}

let (account, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };

let amount = account.data.amount;
account.data.amount = new_amount;
```

### Batch Slurping

Slurp N accounts of the same type—single pointer cast:

```rust
let (accounts, iter) = unsafe { iter.static_slurp_typed_accounts::<TokenAccount, 3>() };
for acc in accounts.iter() {
    process(acc.data.amount);
}
```

### Requirements

- `T: Pod + Zeroable` (bytemuck)
- `size_of::<T>() % 8 == 0` — 8-byte alignment
- `data_len == size_of::<T>()` for each account
- No duplicate markers in the range

## Safety Model

The unsafe methods assume you know the account layout. Slurp wrong type = valid pointer to garbage.

Typical pattern—verify authority first, then trust layout:

```rust
let mut iter = unsafe { AccountIterator::new_from_instruction(input) };

let (authority, iter) = unsafe { iter.known_next_full_account() };
if !(is_program_authority(&authority.static_data.key) && authority.static_data.is_signer()) {
    return Err(ProgramError::Unauthorized);
}

// Authority verified, trust the rest
let (token, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
let (vault, iter) = unsafe { iter.static_slurp_typed_account::<VaultAccount>() };
```

Debug builds verify invariants (counts, no unexpected dups, sizes). Release builds skip checks.

## Helper Methods

`NonDupAccountStatic` provides convenience methods:

```rust
acc.static_data.is_signer()    // -> bool
acc.static_data.is_writable()  // -> bool
acc.static_data.is_executable() // -> bool
```

## Buffer Layout

```text
┌─────────────────────────────────────────────┐
│ num_accounts: u64                           │
├─────────────────────────────────────────────┤
│ Account 0                                   │
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ instruction_data_len: u64                   │
├─────────────────────────────────────────────┤
│ instruction_data: [u8]                      │
├─────────────────────────────────────────────┤
│ program_id: Pubkey (32 bytes)               │
└─────────────────────────────────────────────┘
```

Each account starts with a marker byte: `0xFF` = real account, `0x00-0xFE` = duplicate (index of original).

Real account layout:

```text
┌──────────────────────────────────────────────┐
│ NonDupAccountStatic (128 bytes)              │
│   is_dup, is_signer, is_writable, executable │
│   original_data_len, key, owner              │
│   lamports, data_len                         │
├──────────────────────────────────────────────┤
│ data: [u8; data_len]                         │
├──────────────────────────────────────────────┤
│ _buffer: [u8; 10240] (realloc reserve)       │
├──────────────────────────────────────────────┤
│ padding to 8-byte alignment                  │
├──────────────────────────────────────────────┤
│ rent_epoch: u64                              │
└──────────────────────────────────────────────┘
```

## Types

```rust
pub enum NextAccount {
    Account(AccountInInstruction, AccountIterator),
    Data(&'static mut [u8], &'static Pubkey),  // (instruction_data, program_id)
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
    // ... 10KB buffer, rent_epoch
}
```

## pubkey_byte_map

O(1) pubkey lookup for known key sets. Indexes by first byte, so all keys must have unique first bytes.

```rust
use peephole::pubkey_byte_map::{pubkey_byte_map, PubkeyMap};

const AUTHORITIES: PubkeyMap = pubkey_byte_map(&[ADMIN_KEY, OPERATOR_KEY]);

if !AUTHORITIES.contains(&signer_key) {
    return Err(Unauthorized);
}
```

## assume!

`debug_assert!` in debug builds, `unreachable_unchecked` in release. Use when you know a condition is true but want debug-time verification:

```rust
use peephole::assume;

unsafe {
    assume!(account_count > 0, "expected at least one account");
}
