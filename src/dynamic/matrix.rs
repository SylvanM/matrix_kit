use crate::index;
use std::{ops::{Index, IndexMut}, os::unix::raw::uid_t};

mod constructors;
mod fmt;
mod iter;
mod linalg;
mod ops;
mod shape;

#[cfg(test)]
mod test_util;

// MARK: Matrix Type

/// A dynamically sized matrix with entries of a type `T`
/// 
/// Data is stored column-wise, so adjacent elements of the
/// flatmap vector are in the same column (modulo column breaks)
/// 
/// If T is a Ring, many more useful operations open up. When T is an abstract 
/// data type, this is pretty much only useful for storage.
#[derive(Clone)]
pub struct Matrix<T: Clone> {
	flatmap: Vec<T>,
	row_count: usize,
	col_count: usize
}


#[macro_export]
macro_rules! compatible_vectors {
	($a: expr, $b: expr) => {
		$a.is_vector() && $b.is_vector() && ($a.row_count() == $b.row_count())
	};
}

impl<T: Clone> Matrix<T> {

	// MARK: Constructors

	/// Constructs a matrix from a flat vector
	pub fn from_flatmap(rows: usize, cols: usize, flatmap: Vec<T>) -> Matrix<T> {
		Matrix { flatmap, row_count: rows, col_count: cols }
	}

	/// Constructs a matrix defined index-wise
	pub fn from_index_def(
		rows: usize,
		cols: usize,
		at_index: &mut dyn FnMut(usize, usize) -> T) -> Matrix<T> {

		Matrix::from_flatmap(rows, cols, Vec::from_iter((0..(rows * cols)).map(|i|
			at_index(i % rows, i / rows)
		)))
	}

	// MARK: Properties

	/// Returns whether or not this is a square matrix
	pub fn is_square(&self) -> bool {
		self.row_count() == self.col_count()
	}

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

	/// Returns a copy of the underlying vector of this matrix
	pub fn as_vec(&self) -> Vec<T> {
		self.flatmap.clone()
	}

	/// Returns `true` if this matrix is really just a column vector
	pub fn is_vector(&self) -> bool {
		self.col_count() == 1
	}

	/// Returns whether or not this is a row vector
	pub fn is_row_vector(&self) -> bool {
		self.row_count() == 1
	}

	/// Returns a copy of the entry at row `r` and column `c`
	pub fn get(&self, r: usize, c: usize) -> T {
		debug_assert!(r < self.row_count());
		debug_assert!(c < self.col_count());
		self.flatmap[index!(self.row_count, self.col_count, r, c)].clone()
	}

	/// Sets the entry at row `r` and column `c` to `x`
	pub fn set(&mut self, r: usize, c: usize, x: T) {
		debug_assert!(r < self.row_count());
		debug_assert!(c < self.col_count());
		self.flatmap[index!(self.row_count, self.col_count, r, c)] = x;
	} 
	
}

// MARK: Index

impl<T: Clone> Index<usize> for Matrix<T> {
	type Output = [T];

	/// Returns the column `index` as a slice
	fn index(&self, index: usize) -> &Self::Output {
		&self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

impl<T: Clone> IndexMut<usize> for Matrix<T> {

	/// Returns a mutable reference to the column `index`
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

impl<T: Clone> Index<(usize, usize)> for Matrix<T> {
	type Output = T;

	fn index(&self, index: (usize, usize)) -> &Self::Output {
		let (r, c) = index;
		&self.flatmap[index!(
			self.row_count(),
			self.col_count(),
			r, c
		)]
	}
}

impl<T: Clone> IndexMut<(usize, usize)> for Matrix<T> {
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		let (r, c) = index;
		let m = self.row_count();
		let n = self.col_count();
		&mut self.flatmap[index!(m, n, r, c)]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_index_def() {
		let a = Matrix::from_index_def(10, 9, &mut {|r, c|
			if r == c { 1 } else {0}
		} );

		println!("{:?}", a.transpose());
	}

}
