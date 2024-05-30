//
// A matrix!
//

use std::{cell::RefCell, default, fmt::Debug, mem::swap, ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};

use crate::algebra::*;

/**
 * Translates a row-column coordinate to a flat index in the one dimensional array,
 * indexing along columns. (So, columns are stored next to each other in memory instead of rows)
 */
#[macro_export]
macro_rules! index {
	($m: expr, $n: expr, $r: expr, $c: expr) => {
		$c * $m + $r
	};
}

/// Computes the dot product between the r-th row of the matrix x, and the vector y.
/// 
/// * `M` - 
fn vec_dot_prod_ptr<const M: usize, const N: usize, R: Ring>(x: &[R], r: usize, y: &[R], out: &mut R) {
	*out = R::zero();
	for i in 0..N {
		*out += x[index!(M, N, r, i)] * y[i]
	}
}

fn mat_vec_mul_ptr<const M: usize, const N: usize, R: Ring>(a: &[R], vec: &[R], to_vec: &mut [R]) {
	for r in 0..M {
		vec_dot_prod_ptr::<M, N, R>(a, r, vec, &mut to_vec[r]);
	}
}

fn mat_add<const M: usize, const N: usize, R: Ring>(a: &[R], b: &[R], out: &mut [R]) {
	for i in 0..(M * N) {
		out[i] = a[i] + b[i];
	}
}

fn scalar_mul<const N: usize, R: Ring>(k: R, v: &[R], out: &mut [R]) {
	for i in 0..N {
		out[i] = k * v[i];
	}
}

fn mat_mul_ptrs<const M: usize, const K: usize, const N: usize, R: Ring>(a: &[R], b: &[R], out: &mut [R]) {
	for c in 0..N {
		mat_vec_mul_ptr::<M, K, R>(a, 
			&b[index!(K, N, 0, c)..index!(K, N, K, c)], 
			&mut out[index!(M, N, 0, c)..index!(M, N, M, c)]
		);
	}
}

#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_mat_vec_mul_ptr() {
		// test the super simple identity!
		let identity = [1.into(), 0.into(), 0.into(), 1.into()];
		let simple_vector = [3.into(), 7.into()];
		let mut out_vector = [0.into() ; 2];
		mat_vec_mul_ptr::<2, 2, ZM<11>>(&identity, &simple_vector, &mut out_vector);

		assert_eq!(out_vector, simple_vector);

		let mat = [3.into(), 2.into(), 5.into(), 1.into(), 7.into(), 0.into()];
		let vec = [1.into(), 4.into(), 9.into()];
		let mut out = [0.into() ; 2];
		mat_vec_mul_ptr::<2, 3, ZM<11>>(&mat, &vec, &mut out);

		assert_eq!(out, [9.into(), 6.into()]);

	}

	#[test]
	fn test_full_mat_mul() {
		let a = [4.into(), 8.into(), 5.into(), 5.into(), 6.into(), 5.into(), 9.into(), 1.into(), 10.into(), 2.into(), 1.into(), 0.into()];
		let b = [1.into(), 2.into(), 8.into(), 3.into(), 1.into(), 4.into(), 0.into(), 7.into()];

		let mut prod = [0.into() ; 3 * 2];
		mat_mul_ptrs::<3, 4, 2, ZM<11>>(&a, &b, &mut prod);
		

		assert_eq!(prod, [4.into(), 9.into(), 7.into(), 5.into(), 6.into(), 3.into()]);
	}
}

// MARK: Matrix Type

#[derive(Clone, Copy)]
pub struct Matrix<const M: usize, const N: usize, R: Ring> where [R; M * N]: Sized  {
	pub flatmap: [R ; M * N]
}

impl<const M: usize, const N: usize, R: Ring> Matrix<M, N, R> where [(); M * N]: Sized  {

	pub fn new() -> Self {
		Matrix::from_flatmap([R::zero() ; M * N])
	}

	pub fn from_flatmap(flatmap: [R ; M * N]) -> Self {
		Matrix { flatmap }
	} 

	pub fn identity() -> Self {
		let mut mat = Matrix::new();

		for r in 0..N {
			for c in 0..N {
				if r == c {
					mat[r][c] = R::one();
				}
			}
		}

		mat
	}

	pub fn rhs_concat<const K: usize>(self, rhs: Matrix<M, K, R>) -> Matrix<M, {N + K}, R> where [() ; M * K]: Sized, [() ; M * {N + K}]: Sized {
		let mut big_mat = Matrix::<M, {N + K}, R>::new();

		for r in 0..M {
			for c in 0..N {
				big_mat.flatmap[index!(M, N + K, r, c)] = self.flatmap[index!(M, N, r, c)]
			}
		}

		for r in 0..M {
			for c in 0..K {
				big_mat.flatmap[index!(M, N + K, r, c + N)] = rhs.flatmap[index!(M, K, r, c)]
			}
		}
		
		big_mat
	}

	// MARK: Row Operations

	pub fn swap_rows(mut self, row1: usize, row2: usize) {
		for c in 0..N {
			let temp = self.flatmap[index!(M, N, row1, c)];
			self.flatmap[index!(M, N, row1, c)] = self.flatmap[index!(M, N, row2, c)];
			self.flatmap[index!(M, N, row2, c)] = temp;
		}
	}

	pub fn scale_row(&mut self, row: usize, scalar: R) {
		for c in 0..N {
			self.flatmap[index!(M, N, row, c)] *= scalar
		}
	}

	pub fn add_scaled_row(&mut self, scalar: R, source_row: usize, des_row: usize) {
		for c in 0..N {
			self.flatmap[index!(M, N, des_row, c)] += scalar * self.flatmap[index!(M, N, source_row, c)]
		}
	}

}

// MARK: Index

impl<const M: usize, const N: usize, R: Ring> Index<usize> for Matrix<M, N, R> where [() ; M * N]: Sized {
	type Output = [R];

	/**
	 * Indexes a single COLUMN of this matrix
	 */
	fn index(&self, index: usize) -> &Self::Output {
		&self.flatmap[index!(M, N, 0, index)..index!(M, N, M, index)]
	}
}

impl<const M: usize, const N: usize, R: Ring> IndexMut<usize> for Matrix<M, N, R> where [() ; M * N]: Sized {
	
	/**
	 * Indexes a single row of this matrix
	 */
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[index!(M, N, 0, index)..index!(M, N, M, index)]
	}
}

// MARK: Row and Column Utility
impl<const M: usize, const N: usize, R: Ring> Matrix<M, N, R> where [() ; M * N]: Sized {

	pub fn omitting_row(self, row: usize) -> Matrix<{M - 1}, N, R> where [() ; (M - 1) * N]: Sized {
		let mut row_minor = Matrix::<{M - 1}, N, R>::new();

		let mut r_d = 0;
		for r_s in 0..M {
			if r_s == row { continue; }

			for c in 0..N {
				row_minor.flatmap[index!(M - 1, N, r_d, c)] = self.flatmap[index!(M, N, r_s, c)];
			}

			r_d += 1;
		}

		row_minor
	}

	pub fn omitting_col(self, col: usize) -> Matrix<M, {N - 1}, R> where [() ; M * (N - 1)]: Sized {
		let mut col_minor = Matrix::<M, {N - 1}, R>::new();

		let mut c_d = 0;
		for c_s in 0..N {
			if c_s == col { continue; }

			for r in 0..N {
				col_minor.flatmap[index!(M - 1, N, r, c_d)] = self.flatmap[index!(M, N, r, c_s)];
			}

			c_d += 1;
		}

		col_minor
	}

	pub fn minor(self, row: usize, col: usize) -> Matrix<{M - 1}, {N - 1}, R> where [() ; (M - 1) * (N - 1)]: Sized, {
		let mut minor = Matrix::<{M - 1}, {N - 1}, R>::new();

		let mut r_d = 0;
		for r_s in 0..M {
			if r_s == row { continue; }

			let mut c_d = 0;
			for c_s in 0..N {
				if c_s == col { continue; }
				minor.flatmap[index!(M - 1, N, r_d, c_d)] = self.flatmap[index!(M, N, r_s, c_s)];
				c_d += 1;
			}

			r_d += 1;
		}

		minor
	}

}

// MARK: Operations

impl<const M: usize, const N: usize, R: Ring> Add<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		let mut sum = Matrix::new();
		mat_add::<M, N, R>(&self.flatmap, &rhs.flatmap, &mut sum.flatmap);
		sum
	}
}

impl<const M: usize, const N: usize, R: Ring> AddAssign<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..(M * N) {
			self.flatmap[i] += rhs.flatmap[i]
		}
	}
}

impl<const M: usize, const N: usize, R: Ring> Sub<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		let mut sum = Matrix::new();

		for i in 0..(M * N) {
			sum.flatmap[i] = self.flatmap[i] - rhs.flatmap[i]
		}

		sum
	}
}

impl<const M: usize, const N: usize, R: Ring> SubAssign<Matrix<M, N, R>> for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn sub_assign(&mut self, rhs: Self) {
		for i in 0..(M * N) {
			self.flatmap[i] -= rhs.flatmap[i]
		}
	}
}

impl<const M: usize, const K: usize, const N: usize, R: Ring> Mul<Matrix<K, N, R>> for Matrix<M, K, R> where [R ; M * K]: Sized, [R ; K * N]: Sized, [R ; M * N]: Sized,  {
	type Output = Matrix<M, N, R>;

	fn mul(self, rhs: Matrix<K, N, R>) -> Self::Output {
		let mut prod = Matrix::<M, N, R>::new();
		mat_mul_ptrs::<M, K, N, R>(&self.flatmap, &rhs.flatmap, &mut prod.flatmap);
		prod
	}
}

impl<const M: usize, const N: usize, R: Ring> Neg for Matrix<M, N, R> where [R ; M * N]: Sized, <R as Neg>::Output: Ring {
	type Output = Self;

	fn neg(self) -> Self::Output {
		let mut negative = Matrix::new();
		scalar_mul::<{M * N}, R>(-R::one(), &self.flatmap, &mut negative.flatmap);
		negative
	}
}

impl<const M: usize, const N: usize, R: Ring> Mul<R> for Matrix<M, N, R> where [R ; M * N]: Sized {
	type Output = Matrix<M, N, R>;

	fn mul(self, rhs: R) -> Self::Output {
		let mut scaled = self;
		scaled *= rhs;
		scaled
	}
}

impl<const M: usize, const N: usize, R: Ring> MulAssign<R> for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn mul_assign(&mut self, rhs: R) {
		for i in 0..(M * N) {
			self.flatmap[i] *= rhs;
		}
	}
}

impl<const M: usize, const N: usize, R: Ring> PartialEq for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn eq(&self, other: &Self) -> bool {
		for i in 0..(M * N) {
			if self.flatmap[i] != other.flatmap[i] {
				return false;
			}
		}
		return true;
	}

	fn ne(&self, other: &Self) -> bool {
		for i in 0..(M * N) {
			if self.flatmap[i] == other.flatmap[i] {
				return true;
			}
		}
		return false;
	}
}

impl<const M: usize, const N: usize, R: Ring> Debug for Matrix<M, N, R> where [R ; M * N]: Sized {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		// get the widest value so we know how many spaces we need!
		let mut widest_str_len = 0;
		for i in 0..(M * N) {
			let as_str = format!("{:?}", self.flatmap[i]);
			let this_len = as_str.chars().count();
			if this_len > widest_str_len {
				widest_str_len = this_len;
			}
		}

		// now, we make a vector of strings!
		let mut lines = Vec::<String>::new();

		for r in 0..M {
			let mut this_line = Vec::<String>::new();

			if r == 0 {
				this_line.push("┌ ".to_string());
			} else if r == M - 1 {
				this_line.push("└ ".to_string())
			} else {
				this_line.push("│ ".to_string())
			}

			for c in 0..N {
				let this_entry_str = format!("{:?}", self.flatmap[index!(M, N, r, c)]);
				let this_entry_len = this_entry_str.chars().count();
				this_line.push(format!("{}{}", this_entry_str, " ".repeat(widest_str_len - this_entry_len + 1)));
			}

			if r == 0 {
				this_line.push("┐ ".to_string());
			} else if r == M - 1 {
				this_line.push("┘ ".to_string())
			} else {
				this_line.push("│ ".to_string())
			}

			lines.push(this_line.join(""));
		}

		write!(f, "\n{}", lines.join("\n"))
	}
}

// MARK: Gaussian Elimination

/**
 * Performs row reduction to echelon form, but **not** necessarily *reduced* row echelon form, This should *only* be called from `rowEchelon`
 * and if this is also being used to compute an inverse, the sequence of row operations perfomed on this matrix
 * are also performed on a given matrix as a recipient.
 * 
 * This implementation is ONLY for rings. If you are working in a field, don't use this! If you are working in a ring 
 * and wish to do row reduction, you run the risk of overflow since the entries grow huge, since we can't do any sort 
 * of normalization.
 */ 
fn _ring_ref_rec<const M: usize, const N: usize, R: Ring>(matrix: &mut Matrix<M, N, R>, pivots: &mut [usize ; N], starting_col: usize, pivot_row: usize, scale_tracker: &mut R, swap_count: &mut i32) where [(); M * N]: Sized {

	if pivot_row == M || starting_col == N {
		if starting_col != N {
			// -1 out the remaining pivots
			for c in starting_col..N {
				pivots[c] = usize::MAX; // hopefully no matrix is big enough that this is a problem
			}
		}

		return;
	}

	pivots[starting_col] = pivot_row;

	if matrix.flatmap[index!(M, N, pivot_row, starting_col)] == R::zero() {
		// find the first nonzero entry of this column, and move that to be the pivot
		for r in (pivot_row + 1)..M {
			if matrix.flatmap[index!(M, N, r, starting_col)] != R::zero() {
				// we found a new pivot possibility! Swap it to be the pivot.
				matrix.swap_rows(pivot_row, r);
				*swap_count += 1;
				_ring_ref_rec(matrix, pivots, starting_col, pivot_row, scale_tracker, swap_count);
				return;
			}
		}

		// there are only zeros from here down, so this entry is not a pivot.
		pivots[starting_col] = usize::MAX;
		_ring_ref_rec(matrix, pivots, starting_col + 1, pivot_row, scale_tracker, swap_count);
		return;
	}


	let pivot_entry = matrix.flatmap[index!(M, N, pivot_row, starting_col)];
        
	for r in (pivot_row + 1)..M {
		let entry = matrix.flatmap[index!(M, N, r, starting_col)];
		if entry == R::zero() { continue; }
		
		matrix.scale_row(r, pivot_entry);
		*scale_tracker *= pivot_entry;

		matrix.add_scaled_row(-entry, pivot_row, r);
	}

	_ring_ref_rec(matrix, pivots, starting_col + 1, pivot_row + 1, scale_tracker, swap_count)
}

/**
 * Performs row reduction for matrices with entries in a *FIELD*
 */
fn _field_ref_rec<const M: usize, const N: usize, F: Field>(matrix: &mut Matrix<M, N, F>, pivots: &mut [usize ; N], starting_col: usize, pivot_row: usize, scale_tracker: &mut F, swap_count: &mut i32) where [(); M * N]: Sized {

	if pivot_row == M || starting_col == N {
		if starting_col != N {
			// -1 out the remaining pivots
			for c in starting_col..N {
				pivots[c] = usize::MAX; // hopefully no matrix is big enough that this is a problem
			}
		}

		return;
	}

	pivots[starting_col] = pivot_row;

	if matrix.flatmap[index!(M, N, pivot_row, starting_col)] == F::zero() {
		// find the first nonzero entry of this column, and move that to be the pivot
		for r in (pivot_row + 1)..M {
			if matrix.flatmap[index!(M, N, r, starting_col)] != F::zero() {
				// we found a new pivot possibility! Swap it to be the pivot.
				matrix.swap_rows(pivot_row, r);
				*swap_count += 1;
				_field_ref_rec(matrix, pivots, starting_col, pivot_row, scale_tracker, swap_count);
				return;
			}
		}

		// there are only zeros from here down, so this entry is not a pivot.
		pivots[starting_col] = usize::MAX;
		_field_ref_rec(matrix, pivots, starting_col + 1, pivot_row, scale_tracker, swap_count);
		return;
	}


	let pivot_entry = matrix.flatmap[index!(M, N, pivot_row, starting_col)];
        
	for r in (pivot_row + 1)..M {
		let entry = matrix.flatmap[index!(M, N, r, starting_col)];
		if entry == F::zero() { continue; }

		let scalar = -entry / pivot_entry;

		matrix.add_scaled_row(scalar, pivot_row, r);
	}

	_field_ref_rec(matrix, pivots, starting_col + 1, pivot_row + 1, scale_tracker, swap_count)
}

/**
 * This function is *only* to be called by reduced_ref(), only after ref() has been called.
 */
fn _rref_rec<const M: usize, const N: usize, F: Field>(matrix: &mut Matrix<M, N, F>, pivots: &mut [usize ; N], starting_col: usize) where [(); M * N]: Sized {
	
	if starting_col == N {
		return; // we're done
	}

	let pivot_row = pivots[starting_col];

	if pivot_row == usize::MAX {
		// skip this column
		_rref_rec(matrix, pivots, starting_col + 1);
	} else {
		// normalize this row relative to the pivot
		let pivot_entry = matrix.flatmap[index!(M, N, pivot_row, starting_col)];
		matrix.scale_row(pivot_row, pivot_entry.inverse());

		// eliminate other entries in this column above
		for r in 0..pivot_row {

			let entry = matrix.flatmap[index!(M, N, r, starting_col)];

			if entry == F::zero() { continue; }

			matrix.add_scaled_row(-entry, pivot_row, r);
		}

		_rref_rec(matrix, pivots, starting_col + 1);
	}
}

pub trait RingREF {
	fn row_echelon_inplace(&mut self);
	fn row_echelon(self) -> Self;
}

pub trait FieldREF: RingREF {
	fn reduced_row_echelon_inplace(&mut self);
	fn reduced_row_echelon(self) -> Self;
}

/**
 * Standard implementations of REF for Ring matrices
 */
impl<const M: usize, const N: usize, R: Ring> RingREF for Matrix<M, N, R> where [(); M * N]: Sized {
	default fn row_echelon_inplace(&mut self) {
		println!("Doing REF for a matrix with entries in a RING");
		
		let mut pivots = [0 ; N];
		let mut st = R::one();
		let mut sc = 9;
		_ring_ref_rec(self, &mut pivots, 0, 0, &mut st, &mut sc);
	}

	default fn row_echelon(mut self) -> Self {
		self.row_echelon_inplace();
		self
	}
}

impl<const M: usize, const N: usize, F: Field> RingREF for Matrix<M, N, F> where [(); M * N]: Sized {
	fn row_echelon_inplace(&mut self) {
		println!("Doing REF for a matrix with entries in a FIELD");

		let mut pivots = [0 ; N];
		let mut st = F::one();
		let mut sc = 0;
		_field_ref_rec(self, &mut pivots, 0, 0, &mut st, &mut sc);
	}

	fn row_echelon(mut self) -> Self {
		self.row_echelon_inplace();
		self
	}
}

impl<const M: usize, const N: usize, F: Field> FieldREF for Matrix<M, N, F> where [(); M * N]: Sized {
	fn reduced_row_echelon_inplace(&mut self) {
		println!("Doing RREF for a matrix with entries in a FIELD");
		
		let mut pivots = [0 ; N];
		let mut st = F::one();
		let mut sc = 0;
		_field_ref_rec(self, &mut pivots, 0, 0, &mut st, &mut sc);
		_rref_rec(self, &mut pivots, 0);
	}

	fn reduced_row_echelon(mut self) -> Self {
		self.reduced_row_echelon_inplace();
		self
	}
}

// MARK: Determinant

pub trait Det<R: Ring> {
	fn det(self) -> R;
}

/**
 * Standard recursive cofactor expansion, done the good old-fashioned way. This is NOT efficient for large matrices.
 */
fn _rec_cofactor_det<R: Ring>(flatmap: &[R], n: usize) -> R {
	if n == 1 {
		flatmap[0]
	}
	else if n == 2 {
		// just do simple formula for determinant, 2x2 matrices are pretty common
		flatmap[index!(n, n, 0, 0)] * flatmap[index!(n, n, 1, 1)] - flatmap[index!(n, n, 1, 0)] * flatmap[index!(n, n, 0, 1)]
	} else {
		// we'll just do cofactor expansion along the first row
		let mut determinant = R::zero();
		for col in 0..n {
			// compute determinant of the minor matrix here. So, we construct a new matrix that skips over column c.
			let mut minor_vector = Vec::<R>::with_capacity((n - 1) * (n - 1));

			for c in 0..n {
				if c == col { continue; }
				for r in 1..n {
					minor_vector.push(flatmap[index!(n, n, r, c)]);
				}
			}

			let sub_determinant = _rec_cofactor_det(minor_vector.as_mut_slice(), n - 1);
			determinant += flatmap[index!(n, n, 0, col)] * (if col % 2 == 0 { sub_determinant } else { -sub_determinant });
		}
		determinant
	}
}

impl<const N: usize, R: Ring> Det<R> for Matrix<N, N, R> where [() ; N * N]: Sized {
	default fn det(self) -> R {
		println!("Computing determinant for a RING matrix");
		_rec_cofactor_det(&self.flatmap, N)
	}
}

impl<const N: usize, F: Field> Det<F> for Matrix<N, N, F> where [() ; N * N]: Sized {
	fn det(mut self) -> F {
		let mut swap_count = 0;
		let mut scale_tracker = F::one();
		let mut pivots = [0 ; N];
		_field_ref_rec(&mut self, &mut pivots, 0, 0, &mut scale_tracker, &mut swap_count);
		
		let mut determinant = F::one();

		for i in 0..N {
			determinant *= self.flatmap[index!(N, N, i, i)];
		}

		if swap_count % 2 == 0 {
			determinant
		} else {
			-determinant
		}
	}
}

// MARK: Vectors

impl<const N: usize, R: Ring> InnerProductSpace<R> for Matrix<N, 1, R> where [() ; N * 1]: Sized {
	fn inner_product(self, other: Self) -> R {
		let mut prod = R::zero();
		vec_dot_prod_ptr::<1, N, R>(&self.flatmap, 0, &other.flatmap, &mut prod);
		prod
	}
}

impl<const N: usize> NormSpace for Matrix<N, 1, f64> where [() ; N * 1]: Sized {
	fn norm(self) -> f64 {
		f64::sqrt(self.inner_product(self))
	}
}