//
// A matrix!
//

use std::{fmt::Debug, ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};

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
 */ 
fn _ref_rec<const M: usize, const N: usize, F: Field>(matrix: &mut Matrix<M, N, F>, pivots: &mut [usize ; N], starting_col: usize, pivot_row: usize) where [(); M * N]: Sized {

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
				_ref_rec(matrix, pivots, starting_col, pivot_row);
				return;
			}
		}

		// there are only zeros from here down, so this entry is not a pivot.
		pivots[starting_col] = usize::MAX;
		_ref_rec(matrix, pivots, starting_col + 1, pivot_row);
		return;
	}


	let pivot_entry = matrix.flatmap[index!(M, N, pivot_row, starting_col)];
        
	for r in (pivot_row + 1)..M {
		let entry = matrix.flatmap[index!(M, N, r, starting_col)];
		if entry == F::zero() { continue; }

		let scalar = -entry / pivot_entry;

		matrix.add_scaled_row(scalar, pivot_row, r);
	}

	_ref_rec(matrix, pivots, starting_col + 1, pivot_row + 1)
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

impl<const M: usize, const N: usize, F: Field> Matrix<M, N, F> where [(); M * N]: Sized {

	/**
	 * Converts this matrix into row echelon form, in-place.
	 *
	 * Different authors use different meanings of "row echelon form" versus "*reduced* row echelon form", so for clarity,
     * I am using the same definitions as are used here: https://en.wikipedia.org/wiki/Row_echelon_form
	 */
	pub fn row_echelon(&mut self) {
		let mut pivots = [0 ; N];
		_ref_rec(self, &mut pivots, 0, 0);
	}

	/**
	 * Converts this matrix into *reduced* row echelon form, in-place.
	 *
	 * Different authors use different meanings of "row echelon form" versus "*reduced* row echelon form", so for clarity,
     * I am using the same definitions as are used here: https://en.wikipedia.org/wiki/Row_echelon_form
	 */
	pub fn reduced_row_echelon(&mut self) {
		let mut pivots = [0 ; N];
		_ref_rec(self, &mut pivots, 0, 0);
		_rref_rec(self, &mut pivots, 0);
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