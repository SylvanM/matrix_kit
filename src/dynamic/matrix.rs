use std::cmp::min;
use std::fmt::Debug;
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
pub struct Matrix<R: Ring> {
	flatmap: Vec<R>,
	row_count: usize,
	col_count: usize
}

impl<R: Ring> Matrix<R> {

	// MARK: Constructors

	/// Constructs a matrix from a flat vector
	pub fn from_flatmap(rows: usize, cols: usize, flatmap: Vec<R>) -> Matrix<R> {
		Matrix { flatmap, row_count: rows, col_count: cols }
	}

	/// Constructs a matrix of all zeroes for a given dimension
	pub fn new(rows: usize, cols: usize) -> Matrix<R> {
		Matrix::from_flatmap(rows, cols, vec![R::zero() ; rows * cols])
	}

	/// Constructs the rows * cols identity matrix
	pub fn identity(rows: usize, cols: usize) -> Matrix<R> {
		let mut mat = Matrix::<R>::new(rows, cols);
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
		at_index: &mut dyn FnMut(usize, usize) -> R) -> Matrix<R> {

		Matrix::from_flatmap(rows, cols, Vec::from_iter((0..rows * cols).map(|i|
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

	/// Returns a copy of the underlying vector of this matrix
	pub fn as_vec(&self) -> Vec<R> {
		self.flatmap.clone()
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

	/// Appends a reference to a column to this matrix on the right
	pub fn append_col_ref(&mut self, col: &mut Vec<R>) { 
		assert_eq!(col.len(), self.row_count());

		self.col_count += 1;
		self.flatmap.append(col);
	}

	/// Appends a reference to a matrix to this matrix on the right
	pub fn append_mat_right_ref(&mut self, mat: &mut Matrix<R>) {
		assert_eq!(self.row_count(), mat.row_count());

		self.col_count += mat.col_count();
		self.flatmap.append(&mut mat.flatmap);
	}

	/// Appends a column to this matrix on the right
	pub fn append_col(&mut self, col: Vec<R>) {
		assert_eq!(col.len(), self.row_count());
		let mut new_flatmap = vec![R::zero() ; self.flatmap.len() + col.len()];
		
		for i in 0..self.flatmap.len() {
			new_flatmap[i] = self.flatmap[i].clone();
		}

		for i in 0..col.len() {
			new_flatmap[i + self.flatmap.len()] = col[i].clone();
		}

		self.col_count += 1;
		self.flatmap = new_flatmap;
	}

	/// Appends a matrix to this matrix, on the right
	pub fn append_mat_right(&mut self, mat: Matrix<R>) {
		assert_eq!(self.row_count(), mat.row_count());
		let mut new_flatmap = vec![R::zero() ; self.flatmap.len() + mat.flatmap.len()];

		for i in 0..self.flatmap.len() {
			new_flatmap[i] = self.flatmap[i].clone();
		}

		for i in 0..mat.flatmap.len() {
			new_flatmap[i + self.flatmap.len()] = mat.flatmap[i].clone();
		}

		self.col_count += mat.col_count();
		self.flatmap = new_flatmap;
	}

	/// Appends a row to this matrix on the bottom
	pub fn append_row(&mut self, row: Vec<R>) {
		assert_eq!(row.len(), self.col_count());
		let mut new_flatmap = vec![R::zero() ; self.flatmap.len() + row.len()];
		
		// Set the original values in the new flatmap!
		for r in 0..self.row_count() {
			for c in 0..self.col_count() {
				new_flatmap[
					index!(self.row_count() + 1, self.col_count(), r, c)
				] = self.flatmap[
					index!(self.row_count(), self.col_count(), r, c)
				].clone();
			}
		}

		// set the new row in the flatmap!
		for c in 0..row.len() {
			new_flatmap[
				index!(
					self.row_count() + 1, 
					self.col_count(), 
					self.row_count(), c
				)
			] = row[c].clone();
		}

		self.row_count += 1;
		self.flatmap = new_flatmap;
	}

	/// Appends a matrix to this matrix, on the right
	pub fn append_mat_bottom(&mut self, mat: Matrix<R>) {
		assert_eq!(self.col_count(), mat.col_count());
		let mut new_flatmap = vec![R::zero() ; self.flatmap.len() + mat.flatmap.len()];

		// Set the original values in the new flatmap!
		for r in 0..self.row_count() {
			for c in 0..self.col_count() {
				new_flatmap[
					index!(self.row_count() + mat.row_count(), self.col_count(), r, c)
				] = self.flatmap[
					index!(self.row_count(), self.col_count(), r, c)
				].clone();
			}
		}

		// Set the new rows!

		for r in 0..mat.row_count() {
			for c in 0..mat.col_count() {
				new_flatmap[
					index!(self.row_count() + mat.row_count(), self.col_count(), r, c)
				] = self.flatmap[
					index!(self.row_count(), self.col_count(), r + self.row_count(), c)
				].clone();
			}
		}

		for i in 0..mat.flatmap.len() {
			new_flatmap[i + self.flatmap.len()] = mat.flatmap[i].clone();
		}

		self.col_count += mat.col_count();
		self.flatmap = new_flatmap;
	}

	// MARK: Utility

	/// Applies a function to all entries in this matrix, returning the result 
	/// as a separate matrix
	pub fn applying_to_all<J: Ring>(&self, f: &dyn Fn(R) -> J) -> Matrix<J> {
		Matrix { 
			flatmap: self.flatmap.iter().map(|x| f(x.clone())).collect(), 
			row_count: self.row_count(), 
			col_count: self.col_count() 
		}
	}

	/// Applies a function to all entries in this matrix, in place
	pub fn apply_to_all(&mut self, f: &dyn Fn(R) -> R) {
		for i in 0..self.flatmap.len() {
			self.flatmap[i] = f(self.flatmap[i].clone())
		}
	}

	/// The transpose of this matrix 
	pub fn transpose(&self) -> Matrix<R> {
		Matrix::from_index_def(self.col_count(), self.row_count, &mut |r, c| self.get(c, r))
	}

	// MARK: Math

	/// The squared L2 norm of this vector
	pub fn l2_norm_squared(&self) -> R {
		let mut squared_norm = R::zero();

		for r in self.flatmap.clone() {
			squared_norm += r.power(2);
		}

		squared_norm
	}

	/// Computes the point-wise product of this and another matrix of the same 
	/// dimension
	pub fn hadamard(&self, other: Matrix<R>) -> Matrix<R> {

		debug_assert_eq!(self.col_count(), other.col_count());
		debug_assert_eq!(self.row_count(), other.row_count());

		let mut hada = self.clone();

		for i in 0..(self.flatmap.len()) { 
			hada.flatmap[i] *= other.flatmap[i].clone()
		}

		hada
	}

}

// MARK: Debug

impl<R: Ring> Debug for Matrix<R> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		// get the widest value so we know how many spaces we need!
		let mut widest_str_len = 0;
		for i in 0..(self.flatmap.len()) {
			let as_str = format!("{:?}", self.flatmap[i]);
			let this_len = as_str.chars().count();
			if this_len > widest_str_len {
				widest_str_len = this_len;
			}
		}

		let m = self.row_count();
		let n = self.col_count();

		// now, we make a vector of strings!
		let mut lines = Vec::<String>::new();

		if m == 1 {

			let mut this_line = Vec::<String>::new();

			this_line.push("[ ".to_string());

			for c in 0..n {
				let this_entry_str = format!("{:?}", self.flatmap[index!(m, n, 0, c)]);
				let this_entry_len = this_entry_str.chars().count();
				this_line.push(format!("{}{}", this_entry_str, " ".repeat(widest_str_len - this_entry_len + 1)));
			}

			this_line.push("]".to_string());

			lines.push(this_line.join(""));
			
		} else {
			for r in 0..m {
				let mut this_line = Vec::<String>::new();
	
				if r == 0 {
					this_line.push("┌ ".to_string());
				} else if r == m - 1 {
					this_line.push("└ ".to_string())
				} else {
					this_line.push("│ ".to_string())
				}
	
				for c in 0..n {
					let this_entry_str = format!("{:?}", self.flatmap[index!(m, n, r, c)]);
					let this_entry_len = this_entry_str.chars().count();
					this_line.push(format!("{}{}", this_entry_str, " ".repeat(widest_str_len - this_entry_len + 1)));
				}
	
				if r == 0 {
					this_line.push("┐ ".to_string());
				} else if r == m - 1 {
					this_line.push("┘ ".to_string())
				} else {
					this_line.push("│ ".to_string())
				}
	
				lines.push(this_line.join(""));
			}
		}
		

		write!(f, "\n{}", lines.join("\n"))
	}
}

// MARK: Index

impl<R: Ring> Index<usize> for Matrix<R> {
	type Output = [R];

	/// Returns the column `index` as a slice
	fn index(&self, index: usize) -> &Self::Output {
		&self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

impl<R: Ring> IndexMut<usize> for Matrix<R> {

	/// Returns a mutable reference to the column `index`
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.flatmap[
			index!(self.row_count, self.col_count, 0, index)..
			index!(self.row_count, self.col_count, self.row_count, index)
		]
	}
}

// MARK: Random Constructors

impl<R: Ring> Matrix<R> where Standard: Distribution<R> {

	/// Constructs a random matrix
	pub fn rand(rows: usize, cols: usize) -> Matrix<R> {
		let mut rng = rand::thread_rng();
		Matrix::from_flatmap(
			rows, cols, 
			(0..rows * cols).map(
				|_| rng.gen()
			).collect()
		)
	}

}


// MARK: Comparison

impl<R: Ring> PartialEq for Matrix<R> {
	fn eq(&self, other: &Self) -> bool {
		self.row_count == other.row_count && self.col_count == other.col_count && self.flatmap == other.flatmap
	}
}


// MARK: Operations

impl<R: Ring> Add for Matrix<R> {
	type Output = Matrix<R>;

	fn add(self, rhs: Matrix<R>) -> Matrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		mat_add(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap,
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> AddAssign for Matrix<R> {
	fn add_assign(&mut self, rhs: Matrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_add_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Sub for Matrix<R> {
	type Output = Matrix<R>;

	fn sub(self, rhs: Matrix<R>) -> Matrix<R> {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		mat_sub(self.row_count, self.col_count, 
			&self.flatmap, &rhs.flatmap, 
			&mut out.flatmap
		);
		out
	}
}

impl<R: Ring> SubAssign for Matrix<R> {
	fn sub_assign(&mut self, rhs: Matrix<R>) {
		assert_eq!(self.row_count, rhs.row_count);
		assert_eq!(self.col_count, rhs.col_count);

		mat_sub_assign(self.row_count, self.col_count, 
			&mut self.flatmap, &rhs.flatmap
		);
	}
}

impl<R: Ring> Mul<R> for Matrix<R> {
	type Output = Matrix<R>;

	fn mul(self, rhs: R) -> Self::Output {
		let mut out = Matrix::<R>::new(self.row_count, self.col_count);
		scalar_mul(self.row_count * self.col_count, 
			rhs, 
			&self.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<R> for Matrix<R> {
	fn mul_assign(&mut self, rhs: R) {
		scalar_mul_assign(self.row_count * self.col_count, 
			rhs, &mut self.flatmap
		);
	}
}

impl<R: Ring> Mul<Matrix<R>> for Matrix<R> {
	type Output = Matrix<R>;

	fn mul(self, rhs: Matrix<R>) -> Self::Output {
		assert_eq!(self.col_count, rhs.row_count);

		let mut out = Matrix::<R>::new(self.row_count, rhs.col_count);
		mat_mul_ptrs::<R>(self.row_count, self.col_count, rhs.col_count, 
			&self.flatmap, &rhs.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<R: Ring> MulAssign<Matrix<R>> for Matrix<R> {

	/// Performs in-place multiplication, only valid on square matrices
	fn mul_assign(&mut self, rhs: Matrix<R>) {
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
		let a = Matrix::from_flatmap(2, 2, vec![1, 2, 3, 4]);
		let b = Matrix::from_flatmap(2, 2, vec![5, 6, 7, 8]);
		let c = a + b;
		assert_eq!(c.flatmap, vec![6, 8, 10, 12]);
	}

}