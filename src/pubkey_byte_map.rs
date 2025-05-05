use crate::solana_export::{pubkey::Pubkey, pubkey_bytes, pubkey_from_array};
use bytemuck::{Pod, Zeroable};

use crate::utils::fast_cmp_pubkey;

// cannot compare constant sized arrays using operator ==
const fn bytes_equal(a: [u8; 32], b: [u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(align(8), C)]
pub struct PubkeyMap(pub [Pubkey; 256]);

impl PubkeyMap {
    /// Returns true if the given public key is contained in the map.
    #[inline]
    pub fn contains(&self, key: &Pubkey) -> bool {
        unsafe {
            // the as_ref and as_bytes calls are not inlined properly lol
            let key_as_bytes: &[u8; 32] = std::mem::transmute(key);
            let first = *key_as_bytes.get_unchecked(0) as usize;
            // I don't trust compiler to elide bounds check
            fast_cmp_pubkey(key, self.0.get_unchecked(first))
        }
    }
}

/// Creates a new `PubkeyMap` from the given slice of public keys,
/// in the existing map
///
/// # Panics
///
/// Panics if `keys` contains more than 256 elements or if any two keys share the same first byte.
pub fn try_pubkey_byte_map(keys: &[Pubkey], key_map: &mut PubkeyMap) -> Result<(), &'static str> {
    if keys.len() > 256 {
        return Err("keys length must be less than 256");
    }

    let zero_bytes: [u8; 32] = [0; 32];
    let zero = pubkey_from_array(zero_bytes);

    for k in &mut key_map.0 {
        *k = zero;
    }

    for (i, key) in keys.iter().enumerate() {
        let key_bytes: [u8; 32] = pubkey_bytes(key);
        let first = key_bytes[0] as usize;
        if bytes_equal(pubkey_bytes(&key_map.0[first]), zero_bytes) {
            key_map.0[first] = *key;
        } else {
            return Err(ERROR_MSGS[i]);
        }
    }

    Ok(())
}

/// Creates a new `PubkeyMap` from the given slice of public keys.
///
/// # Panics
///
/// Panics if `keys` contains more than 256 elements or if any two keys share the same first byte.
#[inline(always)]
pub const fn pubkey_byte_map(keys: &[Pubkey]) -> PubkeyMap {
    if keys.len() > 256 {
        panic!("keys length must be less than 256");
    }

    let zero_bytes: [u8; 32] = [0; 32];
    let zero = pubkey_from_array(zero_bytes);
    let mut key_map = [zero; 256];

    let mut i = 0;

    while i < keys.len() {
        let key = keys[i];
        let key_bytes: [u8; 32] = pubkey_bytes(&key);
        let first = key_bytes[0] as usize;
        if bytes_equal(pubkey_bytes(&key_map[first]), zero_bytes) {
            key_map[first] = key;
        } else {
            panic!("{}", ERROR_MSGS[i]);
        }
        i += 1;
    }

    PubkeyMap(key_map)
}

const ERROR_MSGS: [&str; 256] = [
    "Key 0 shares the same first byte as another key",
    "Key 1 shares the same first byte as another key",
    "Key 2 shares the same first byte as another key",
    "Key 3 shares the same first byte as another key",
    "Key 4 shares the same first byte as another key",
    "Key 5 shares the same first byte as another key",
    "Key 6 shares the same first byte as another key",
    "Key 7 shares the same first byte as another key",
    "Key 8 shares the same first byte as another key",
    "Key 9 shares the same first byte as another key",
    "Key 10 shares the same first byte as another key",
    "Key 11 shares the same first byte as another key",
    "Key 12 shares the same first byte as another key",
    "Key 13 shares the same first byte as another key",
    "Key 14 shares the same first byte as another key",
    "Key 15 shares the same first byte as another key",
    "Key 16 shares the same first byte as another key",
    "Key 17 shares the same first byte as another key",
    "Key 18 shares the same first byte as another key",
    "Key 19 shares the same first byte as another key",
    "Key 20 shares the same first byte as another key",
    "Key 21 shares the same first byte as another key",
    "Key 22 shares the same first byte as another key",
    "Key 23 shares the same first byte as another key",
    "Key 24 shares the same first byte as another key",
    "Key 25 shares the same first byte as another key",
    "Key 26 shares the same first byte as another key",
    "Key 27 shares the same first byte as another key",
    "Key 28 shares the same first byte as another key",
    "Key 29 shares the same first byte as another key",
    "Key 30 shares the same first byte as another key",
    "Key 31 shares the same first byte as another key",
    "Key 32 shares the same first byte as another key",
    "Key 33 shares the same first byte as another key",
    "Key 34 shares the same first byte as another key",
    "Key 35 shares the same first byte as another key",
    "Key 36 shares the same first byte as another key",
    "Key 37 shares the same first byte as another key",
    "Key 38 shares the same first byte as another key",
    "Key 39 shares the same first byte as another key",
    "Key 40 shares the same first byte as another key",
    "Key 41 shares the same first byte as another key",
    "Key 42 shares the same first byte as another key",
    "Key 43 shares the same first byte as another key",
    "Key 44 shares the same first byte as another key",
    "Key 45 shares the same first byte as another key",
    "Key 46 shares the same first byte as another key",
    "Key 47 shares the same first byte as another key",
    "Key 48 shares the same first byte as another key",
    "Key 49 shares the same first byte as another key",
    "Key 50 shares the same first byte as another key",
    "Key 51 shares the same first byte as another key",
    "Key 52 shares the same first byte as another key",
    "Key 53 shares the same first byte as another key",
    "Key 54 shares the same first byte as another key",
    "Key 55 shares the same first byte as another key",
    "Key 56 shares the same first byte as another key",
    "Key 57 shares the same first byte as another key",
    "Key 58 shares the same first byte as another key",
    "Key 59 shares the same first byte as another key",
    "Key 60 shares the same first byte as another key",
    "Key 61 shares the same first byte as another key",
    "Key 62 shares the same first byte as another key",
    "Key 63 shares the same first byte as another key",
    "Key 64 shares the same first byte as another key",
    "Key 65 shares the same first byte as another key",
    "Key 66 shares the same first byte as another key",
    "Key 67 shares the same first byte as another key",
    "Key 68 shares the same first byte as another key",
    "Key 69 shares the same first byte as another key",
    "Key 70 shares the same first byte as another key",
    "Key 71 shares the same first byte as another key",
    "Key 72 shares the same first byte as another key",
    "Key 73 shares the same first byte as another key",
    "Key 74 shares the same first byte as another key",
    "Key 75 shares the same first byte as another key",
    "Key 76 shares the same first byte as another key",
    "Key 77 shares the same first byte as another key",
    "Key 78 shares the same first byte as another key",
    "Key 79 shares the same first byte as another key",
    "Key 80 shares the same first byte as another key",
    "Key 81 shares the same first byte as another key",
    "Key 82 shares the same first byte as another key",
    "Key 83 shares the same first byte as another key",
    "Key 84 shares the same first byte as another key",
    "Key 85 shares the same first byte as another key",
    "Key 86 shares the same first byte as another key",
    "Key 87 shares the same first byte as another key",
    "Key 88 shares the same first byte as another key",
    "Key 89 shares the same first byte as another key",
    "Key 90 shares the same first byte as another key",
    "Key 91 shares the same first byte as another key",
    "Key 92 shares the same first byte as another key",
    "Key 93 shares the same first byte as another key",
    "Key 94 shares the same first byte as another key",
    "Key 95 shares the same first byte as another key",
    "Key 96 shares the same first byte as another key",
    "Key 97 shares the same first byte as another key",
    "Key 98 shares the same first byte as another key",
    "Key 99 shares the same first byte as another key",
    "Key 100 shares the same first byte as another key",
    "Key 101 shares the same first byte as another key",
    "Key 102 shares the same first byte as another key",
    "Key 103 shares the same first byte as another key",
    "Key 104 shares the same first byte as another key",
    "Key 105 shares the same first byte as another key",
    "Key 106 shares the same first byte as another key",
    "Key 107 shares the same first byte as another key",
    "Key 108 shares the same first byte as another key",
    "Key 109 shares the same first byte as another key",
    "Key 110 shares the same first byte as another key",
    "Key 111 shares the same first byte as another key",
    "Key 112 shares the same first byte as another key",
    "Key 113 shares the same first byte as another key",
    "Key 114 shares the same first byte as another key",
    "Key 115 shares the same first byte as another key",
    "Key 116 shares the same first byte as another key",
    "Key 117 shares the same first byte as another key",
    "Key 118 shares the same first byte as another key",
    "Key 119 shares the same first byte as another key",
    "Key 120 shares the same first byte as another key",
    "Key 121 shares the same first byte as another key",
    "Key 122 shares the same first byte as another key",
    "Key 123 shares the same first byte as another key",
    "Key 124 shares the same first byte as another key",
    "Key 125 shares the same first byte as another key",
    "Key 126 shares the same first byte as another key",
    "Key 127 shares the same first byte as another key",
    "Key 128 shares the same first byte as another key",
    "Key 129 shares the same first byte as another key",
    "Key 130 shares the same first byte as another key",
    "Key 131 shares the same first byte as another key",
    "Key 132 shares the same first byte as another key",
    "Key 133 shares the same first byte as another key",
    "Key 134 shares the same first byte as another key",
    "Key 135 shares the same first byte as another key",
    "Key 136 shares the same first byte as another key",
    "Key 137 shares the same first byte as another key",
    "Key 138 shares the same first byte as another key",
    "Key 139 shares the same first byte as another key",
    "Key 140 shares the same first byte as another key",
    "Key 141 shares the same first byte as another key",
    "Key 142 shares the same first byte as another key",
    "Key 143 shares the same first byte as another key",
    "Key 144 shares the same first byte as another key",
    "Key 145 shares the same first byte as another key",
    "Key 146 shares the same first byte as another key",
    "Key 147 shares the same first byte as another key",
    "Key 148 shares the same first byte as another key",
    "Key 149 shares the same first byte as another key",
    "Key 150 shares the same first byte as another key",
    "Key 151 shares the same first byte as another key",
    "Key 152 shares the same first byte as another key",
    "Key 153 shares the same first byte as another key",
    "Key 154 shares the same first byte as another key",
    "Key 155 shares the same first byte as another key",
    "Key 156 shares the same first byte as another key",
    "Key 157 shares the same first byte as another key",
    "Key 158 shares the same first byte as another key",
    "Key 159 shares the same first byte as another key",
    "Key 160 shares the same first byte as another key",
    "Key 161 shares the same first byte as another key",
    "Key 162 shares the same first byte as another key",
    "Key 163 shares the same first byte as another key",
    "Key 164 shares the same first byte as another key",
    "Key 165 shares the same first byte as another key",
    "Key 166 shares the same first byte as another key",
    "Key 167 shares the same first byte as another key",
    "Key 168 shares the same first byte as another key",
    "Key 169 shares the same first byte as another key",
    "Key 170 shares the same first byte as another key",
    "Key 171 shares the same first byte as another key",
    "Key 172 shares the same first byte as another key",
    "Key 173 shares the same first byte as another key",
    "Key 174 shares the same first byte as another key",
    "Key 175 shares the same first byte as another key",
    "Key 176 shares the same first byte as another key",
    "Key 177 shares the same first byte as another key",
    "Key 178 shares the same first byte as another key",
    "Key 179 shares the same first byte as another key",
    "Key 180 shares the same first byte as another key",
    "Key 181 shares the same first byte as another key",
    "Key 182 shares the same first byte as another key",
    "Key 183 shares the same first byte as another key",
    "Key 184 shares the same first byte as another key",
    "Key 185 shares the same first byte as another key",
    "Key 186 shares the same first byte as another key",
    "Key 187 shares the same first byte as another key",
    "Key 188 shares the same first byte as another key",
    "Key 189 shares the same first byte as another key",
    "Key 190 shares the same first byte as another key",
    "Key 191 shares the same first byte as another key",
    "Key 192 shares the same first byte as another key",
    "Key 193 shares the same first byte as another key",
    "Key 194 shares the same first byte as another key",
    "Key 195 shares the same first byte as another key",
    "Key 196 shares the same first byte as another key",
    "Key 197 shares the same first byte as another key",
    "Key 198 shares the same first byte as another key",
    "Key 199 shares the same first byte as another key",
    "Key 200 shares the same first byte as another key",
    "Key 201 shares the same first byte as another key",
    "Key 202 shares the same first byte as another key",
    "Key 203 shares the same first byte as another key",
    "Key 204 shares the same first byte as another key",
    "Key 205 shares the same first byte as another key",
    "Key 206 shares the same first byte as another key",
    "Key 207 shares the same first byte as another key",
    "Key 208 shares the same first byte as another key",
    "Key 209 shares the same first byte as another key",
    "Key 210 shares the same first byte as another key",
    "Key 211 shares the same first byte as another key",
    "Key 212 shares the same first byte as another key",
    "Key 213 shares the same first byte as another key",
    "Key 214 shares the same first byte as another key",
    "Key 215 shares the same first byte as another key",
    "Key 216 shares the same first byte as another key",
    "Key 217 shares the same first byte as another key",
    "Key 218 shares the same first byte as another key",
    "Key 219 shares the same first byte as another key",
    "Key 220 shares the same first byte as another key",
    "Key 221 shares the same first byte as another key",
    "Key 222 shares the same first byte as another key",
    "Key 223 shares the same first byte as another key",
    "Key 224 shares the same first byte as another key",
    "Key 225 shares the same first byte as another key",
    "Key 226 shares the same first byte as another key",
    "Key 227 shares the same first byte as another key",
    "Key 228 shares the same first byte as another key",
    "Key 229 shares the same first byte as another key",
    "Key 230 shares the same first byte as another key",
    "Key 231 shares the same first byte as another key",
    "Key 232 shares the same first byte as another key",
    "Key 233 shares the same first byte as another key",
    "Key 234 shares the same first byte as another key",
    "Key 235 shares the same first byte as another key",
    "Key 236 shares the same first byte as another key",
    "Key 237 shares the same first byte as another key",
    "Key 238 shares the same first byte as another key",
    "Key 239 shares the same first byte as another key",
    "Key 240 shares the same first byte as another key",
    "Key 241 shares the same first byte as another key",
    "Key 242 shares the same first byte as another key",
    "Key 243 shares the same first byte as another key",
    "Key 244 shares the same first byte as another key",
    "Key 245 shares the same first byte as another key",
    "Key 246 shares the same first byte as another key",
    "Key 247 shares the same first byte as another key",
    "Key 248 shares the same first byte as another key",
    "Key 249 shares the same first byte as another key",
    "Key 250 shares the same first byte as another key",
    "Key 251 shares the same first byte as another key",
    "Key 252 shares the same first byte as another key",
    "Key 253 shares the same first byte as another key",
    "Key 254 shares the same first byte as another key",
    "Key 255 shares the same first byte as another key",
];

#[cfg(test)]
mod tests {
    use crate::solana_export::unique_pubkey;

    use super::*;

    #[test]
    fn test_contains() {
        let key1 = pubkey_from_array([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]);
        let key2 = pubkey_from_array([
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
            0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c,
            0x2d, 0x2e, 0x2f, 0x30,
        ]);
        let key3 = pubkey_from_array([
            0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d,
            0x3e, 0x3f, 0x40, 0x41,
        ]);

        let keys = &[key1, key2];
        let map = pubkey_byte_map(keys);

        assert!(map.contains(&key1));
        assert!(map.contains(&key2));
        assert!(!map.contains(&key3));
    }

    #[test]
    #[should_panic(expected = "keys length must be less than 256")]
    fn test_pubkey_byte_map_too_many_keys() {
        let keys: Vec<_> = (0..300).map(|_| unique_pubkey()).collect();
        let _ = pubkey_byte_map(&keys);
    }

    #[test]
    #[should_panic(expected = "Key 1 shares the same first byte as another key")]
    fn test_pubkey_byte_map_duplicate_first_bytes() {
        let key1 = unique_pubkey();
        let keys = &[key1, key1];
        let _ = pubkey_byte_map(keys);
    }

    #[test]
    fn test_pubkey_byte_map_non_matching_keys_with_same_first_byte() {
        let key1 = pubkey_from_array([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]);
        let key2 = pubkey_from_array([
            0x00, 0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3,
            0xf2, 0xf1, 0xf0, 0xef, 0xee, 0xed, 0xec, 0xeb, 0xea, 0xe9, 0xe8, 0xe7, 0xe6, 0xe5,
            0xe4, 0xe3, 0xe2, 0xe1,
        ]);

        let keys = &[key1];
        let map = pubkey_byte_map(keys);

        assert!(map.contains(&key1));
        assert!(!map.contains(&key2));
    }

    #[test]
    fn test_try_pubkey_byte_map() {
        let key1 = pubkey_from_array([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]);
        let key2 = pubkey_from_array([
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
            0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c,
            0x2d, 0x2e, 0x2f, 0x30,
        ]);
        let key3 = pubkey_from_array([
            0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d,
            0x3e, 0x3f, 0x40, 0x41,
        ]);

        let keys = &[key1, key2];
        let mut map = PubkeyMap([Pubkey::default(); 256]);
        assert!(try_pubkey_byte_map(keys, &mut map).is_ok());

        assert!(map.contains(&key1));
        assert!(map.contains(&key2));
        assert!(!map.contains(&key3));
    }

    #[test]
    fn test_try_pubkey_byte_map_too_many_keys() {
        let keys: Vec<_> = (0..300).map(|_| unique_pubkey()).collect();
        let mut map = PubkeyMap([Pubkey::default(); 256]);
        assert_eq!(
            try_pubkey_byte_map(&keys, &mut map),
            Err("keys length must be less than 256")
        );
    }

    #[test]
    fn test_try_pubkey_byte_map_duplicate_first_bytes() {
        let key1 = unique_pubkey();
        let keys = &[key1, key1];
        let mut map = PubkeyMap([Pubkey::default(); 256]);
        assert_eq!(
            try_pubkey_byte_map(keys, &mut map),
            Err("Key 1 shares the same first byte as another key")
        );
    }

    #[test]
    fn test_try_pubkey_byte_map_non_matching_keys_with_same_first_byte() {
        let key1 = pubkey_from_array([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]);
        let key2 = pubkey_from_array([
            0x00, 0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3,
            0xf2, 0xf1, 0xf0, 0xef, 0xee, 0xed, 0xec, 0xeb, 0xea, 0xe9, 0xe8, 0xe7, 0xe6, 0xe5,
            0xe4, 0xe3, 0xe2, 0xe1,
        ]);

        let keys = &[key1];
        let mut map = PubkeyMap([Pubkey::default(); 256]);
        assert!(try_pubkey_byte_map(keys, &mut map).is_ok());

        assert!(map.contains(&key1));
        assert!(!map.contains(&key2));
    }
}
