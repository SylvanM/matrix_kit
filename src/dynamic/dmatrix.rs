use std::cmp::min;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::usize;
use algebra_kit::algebra::Ring;
use rand::Rng;
use rand::distributions::{Distribution, Standard};
use crate::index;
use crate::dynamic::dynamic_vector_util::*;

// MARK: Matrix Type

/// A dynamically sized matrix with entries in a ring `R`
/// 
/// Data is stored column-wise, so adjacent elements of the
/// flatmap vector are in the same column (modulo column breaks)
#[derive(Clone)]
pub struct DMatrix<R: Ring> {
	flatmap: Vec<R>,
	row_count: usize,
	col_count: usize
}

impl<R: Ring> DMatrix<R> {

	// MARK: Constructors

	/// Constructs a matrix from a flat vector
	pub fn from_flatmap(rows: usize, cols: usize, flatmap: Vec<R>) -> DMatrix<R> {
		DMatrix { flatmap, row_count: rows, col_count: cols }
	}

	/// Constructs a matrix of all zeroes for a given dimension
	pub fn new(rows: usize, cols: usize) -> DMatrix<R> {
		DMatrix::from_flatmap(rows, cols, vec![R::zero() ; rows * cols])
	}

	/// Constructs the rows * cols identity matrix
	pub fn identity(rows: usize, cols: usize) -> DMatrix<R> {
		let mut mat = DMatrix::<R>::new(rows, cols);
		let limiting_dimension = min(rows, cols);

		for r in 0..limiting_dimension {
			for c in 0..limiting_dimension {
				mat.flatmap[index!(rows, cols, r, c)] = R::one();
			}
		}

		mat
	}

	/// Constructs a matrix defined index-wise
	pub fn from_index_def(
		rows: usize,
		cols: usize,
		at_index: &dyn Fn(usize, usize) -> R) -> DMatrix<R> {

		DMatrix::from_flatmap(rows, cols, Vec::from_iter((0..rows * cols).map(|i|
			at_index(i / rows, i % rows)
		)))
	}

	// MARK: Properties

	/// The amount of rows in this matrix
	#[inline]
	pub fn row_count(&self) -> usize {
		self.row_count
	}

	/// The amount of columns in this matrix
	#[inline]
	pub fn col_count(&self) -> usize {
		self.col_count
	}

	// MARK: Utility

	/// Returns a copy of the entry at row `r` and column `c`
	pub fn get(&self, r: usize, c: usize) -> R {
		self.flatmap[index!(self.row_count, self.col_count, r, c)].clone()
	}

	/// Sets the entry at row `r` and column `c` to `x`
	pub fn set(&mut self, r: usize, c: usize, x: R) {
		self.flatmap[index!(self.row_count, self.col_count, r, c)] = x;
	} 

}

// MARK: Index

impl<R: Ring> Index<usize> for DMatrix<R> {
	type Output = [R];

	/// Returns the column `index` as a slice
	fn index(&self, index: usize) -> &Self::Output {
		&self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

impl<R: Ring> IndexMut<usize> for DMatrix<R> {

	/// Returns a mutable reference to the column `index`
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

// MARK: Random Constructors

impl<R: Ring> DMatrix<R> where Standard: Distribution<R> {

	/// Constructs a random matrix
	pub fn rand(rows: usize, cols: usize) -> DMatrix<R> {
		let mut rng = rand::thread_rng();
		DMatrix::from_flatmap(
			rows, cols, 
			(0..rows * cols).map(
				|_| rng.gen()
			).collect()
		)
	}

}

// MARK: Operations

impl<R: Ring> Add for DMatrix<R> {
	type Output = DMatrix<R>;

	fn add(self, rhs: DMatrix<R>) -> DMatrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = DMatrix::<R>::new(self.row_count, self.col_count);
		mat_add(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap,
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> AddAssign for DMatrix<R> {
	fn add_assign(&mut self, rhs: DMatrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_add_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Sub for DMatrix<R> {
	type Output = DMatrix<R>;

	fn sub(self, rhs: DMatrix<R>) -> DMatrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = DMatrix::<R>::new(self.row_count, self.col_count);
		mat_sub(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap, 
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> SubAssign for DMatrix<R> {
	fn sub_assign(&mut self, rhs: DMatrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_sub_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Mul<R> for DMatrix<R> {
	type Output = DMatrix<R>;

	fn mul(self, rhs: R) -> Self::Output {
		let mut out = DMatrix::<R>::new(self.row_count, self.col_count);
		scalar_mul(self.row_count * self.col_count, 
			rhs, 
			&self.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<R> for DMatrix<R> {
	fn mul_assign(&mut self, rhs: R) {
		scalar_mul_assign(self.row_count * self.col_count, 
			rhs, &mut self.flatmap
		);
	}
}

impl<R: Ring> Mul<DMatrix<R>> for DMatrix<R> {
	type Output = DMatrix<R>;

	fn mul(self, rhs: DMatrix<R>) -> Self::Output {
		assert_eq!(self.col_count, rhs.row_count);

		let mut out = DMatrix::<R>::new(self.row_count, rhs.col_count);
		mat_mul_ptrs::<R>(self.row_count, self.col_count, rhs.col_count, 
			&self.flatmap, &rhs.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<DMatrix<R>> for DMatrix<R> {

	/// Performs in-place multiplication, only valid on square matrices
	fn mul_assign(&mut self, rhs: DMatrix<R>) {
		assert_eq!(self.col_count, self.row_count); // Only on square matrices!
		assert_eq!(self.col_count, rhs.col_count); // make sure the other matrix is chill
		assert_eq!(self.row_count, rhs.row_count);

		mat_mul_ptrs_assign(self.row_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

#[cfg(test)]
mod matrix_tests {
	use super::*;

	// MARK: Operator tests

	#[test]
	fn test_add() {
		let a = DMatrix::from_flatmap(2, 2, vec![1, 2, 3, 4]);
		let b = DMatrix::from_flatmap(2, 2, vec![5, 6, 7, 8]);
		let c = a + b;
		assert_eq!(c.flatmap, vec![6, 8, 10, 12]);
	}

}