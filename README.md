# `matrix_kit`: An abstract linear algebra library for Rust

# Things to do:
- move SVD stuff from mlkit to here
- re-implement matrix appending to be simpler for all version
- Can't actually initialize empty matrices for non-ring types, e.g. can't just
    have Matrix::<u8>::new(4, 4), this has un-initialized flatmap.
- All the mat ops (adding, multing) should just take references, shouldn't
    need to clone everything constantly.
- Rewrite row_reduce