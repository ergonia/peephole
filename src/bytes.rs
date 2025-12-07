use core::mem::MaybeUninit;

use bytemuck::Pod;

/// Casts a raw pointer to a mutable reference of type T.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - The pointer is properly aligned for T.
/// - The pointer points to an initialized instance of T.
/// - The lifetime of the returned reference does not outlive the pointed-to data.
#[inline(always)]
pub unsafe fn cast_ptr<T: Pod>(input: *mut u8) -> &'static mut T {
    let the_ref;
    #[cfg(debug_assertions)]
    {
        let as_slice = core::slice::from_raw_parts_mut(input, core::mem::size_of::<T>());
        the_ref = bytemuck::from_bytes_mut(as_slice);
    }
    #[cfg(not(debug_assertions))]
    {
        the_ref = &mut *(input as *mut T);
    }
    the_ref
}

/// Casts a raw pointer to a constant reference of type T.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - The pointer is properly aligned for T.
/// - The pointer points to an initialized instance of T.
/// - The lifetime of the returned reference does not outlive the pointed-to data.
#[inline(always)]
pub unsafe fn cast_ptr_const<T: Pod>(input: *const u8) -> &'static T {
    let the_ref;
    #[cfg(debug_assertions)]
    {
        let as_slice = core::slice::from_raw_parts(input, core::mem::size_of::<T>());
        the_ref = bytemuck::from_bytes(as_slice);
    }
    #[cfg(not(debug_assertions))]
    {
        the_ref = &*(input as *const T);
    }
    the_ref
}

/// Reads a value of type T from a raw pointer and returns it along with the next pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - The pointer is properly aligned for T.
/// - The pointer points to an initialized instance of T.
/// - There is enough memory allocated after the pointer to hold T.
/// - The lifetime of the returned reference does not outlive the pointed-to data.
#[inline(always)]
pub unsafe fn slurp<T: Pod>(input: *mut u8) -> (&'static mut T, *mut u8) {
    let next_ptr = input.add(core::mem::size_of::<T>());

    (cast_ptr::<T>(input), next_ptr)
}

/// Reads a value of type T from a raw pointer and returns it along with the next pointer.
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The caller must ensure that:
/// - The pointer is properly aligned for T.
/// - The pointer points to an initialized instance of T.
/// - There is enough memory allocated after the pointer to hold T.
/// - The lifetime of the returned reference does not outlive the pointed-to data.
#[inline(always)]
pub unsafe fn slurp_const<T: Pod>(input: *const u8) -> (&'static T, *const u8) {
    let next_ptr = input.add(core::mem::size_of::<T>());

    (cast_ptr_const::<T>(input), next_ptr)
}

/// A trait for types that can be converted to and from byte slices.
pub trait PodUtils: Pod {
    /// Converts the implementing type to a slice of bytes.
    fn to_bytes(&self) -> &[u8];

    /// Converts the implementing type to a vector of bytes.
    /// Requires the `std` feature.
    #[cfg(any(test, feature = "std"))]
    fn to_vec(&self) -> std::vec::Vec<u8> {
        self.to_bytes().to_vec()
    }

    /// Attempts to create an instance of the implementing type from a slice of bytes.
    ///
    /// Returns `None` if the slice doesn't have the correct length.
    fn try_from_slice(slice: &[u8]) -> Option<Self>;

    #[inline]
    fn from_slice(slice: &[u8]) -> Self {
        Self::try_from_slice(slice).unwrap()
    }

    #[inline]
    fn try_write_to_slice(&self, into: &mut [u8]) -> bool {
        if into.len() != self.to_bytes().len() {
            return false;
        }
        into.copy_from_slice(self.to_bytes());
        true
    }

    #[inline]
    fn write_to_slice(&self, into: &mut [u8]) {
        assert!(self.try_write_to_slice(into))
    }
}

impl<T: Pod> PodUtils for T {
    #[inline]
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    #[inline]
    fn try_from_slice(slice: &[u8]) -> Option<Self> {
        if slice.len() != core::mem::size_of::<Self>() {
            return None;
        }

        let mut uninit: MaybeUninit<T> = MaybeUninit::uninit();
        unsafe {
            core::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                uninit.as_mut_ptr() as *mut u8,
                core::mem::size_of::<Self>(),
            );
            Some(uninit.assume_init())
        }
    }
}

#[inline(always)]
pub const fn offset_after<T: Sized>(base: usize) -> usize {
    let alignment_of_t = core::mem::align_of::<T>();
    let size_of_t = core::mem::size_of::<T>();

    if base % alignment_of_t == 0 {
        base + size_of_t
    } else {
        panic!("Offset after is not aligned");
    }
}

#[inline(always)]
pub const fn validate_aligned<T: Sized>(base: usize) -> usize {
    if base % core::mem::align_of::<T>() == 0 {
        base
    } else {
        panic!("Offset after is not aligned");
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use std::{vec, vec::Vec};
    use bytemuck::Zeroable;

    use super::*;

    #[repr(C)]
    #[derive(Debug, PartialEq, Clone, Copy, Zeroable, Pod)]
    struct TestStruct {
        y: f64,
        x: i64,
    }

    #[test]
    fn test_cast_ptr() {
        let mut data = TestStruct { x: 42, y: 3.143 };
        let ptr = &mut data as *mut TestStruct as *mut u8;

        unsafe {
            let result = cast_ptr::<TestStruct>(ptr);
            assert_eq!(*result, data);
        }
    }

    #[test]
    fn test_cast_ptr_const() {
        let data = TestStruct { x: 42, y: 3.143 };
        let ptr = &data as *const TestStruct as *const u8;

        unsafe {
            let result = cast_ptr_const::<TestStruct>(ptr);
            assert_eq!(*result, data);
        }
    }

    #[test]
    fn test_slurp() {
        let mut data = [TestStruct { x: 42, y: 3.143 }, TestStruct { x: 10, y: 2.5 }];
        let ptr = data.as_mut_ptr() as *mut u8;

        unsafe {
            let (result, next_ptr) = slurp::<TestStruct>(ptr);
            assert_eq!(*result, data[0]);
            assert_eq!(next_ptr, ptr.add(std::mem::size_of::<TestStruct>()));
        }
    }

    #[test]
    fn test_slurp_const() {
        let data = [TestStruct { x: 42, y: 3.143 }, TestStruct { x: 10, y: 2.5 }];
        let ptr = data.as_ptr() as *const u8;

        unsafe {
            let (result, next_ptr) = slurp_const::<TestStruct>(ptr);
            assert_eq!(*result, data[0]);
            assert_eq!(next_ptr, ptr.add(std::mem::size_of::<TestStruct>()));
        }
    }

    #[test]
    fn test_pod_vec() {
        let data = TestStruct { x: 42, y: 3.143 };
        let vec = data.to_vec();

        assert_eq!(vec.len(), std::mem::size_of::<TestStruct>());

        let reconstructed = TestStruct::try_from_slice(&vec).unwrap();
        assert_eq!(reconstructed, data);

        assert_eq!(TestStruct::try_from_slice(&[0u8; 1]), None);
    }
}
