use std::cmp::min;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Sub, SubAssign};
use std::{usize, vec};
use algebra_kit::algebra::{Field, Ring};
use rand_distr::Distribution;
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

#[macro_export]
macro_rules! compatible_vectors {
	($a: expr, $b: expr) => {
		$a.is_vector() && $b.is_vector() && ($a.row_count() == $b.row_count())
	};
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
		let mut mat = Matrix::new(rows, cols);
		let limiting_dimension = min(rows, cols);

		for i in 0..limiting_dimension {
			mat.set(i, i, R::one());
		}

		mat
	}

	/// Creates a diagonal matrix from a given diagonal
	pub fn from_diagonal(diagonal: Vec<R>) -> Matrix<R> {
		Matrix::from_index_def(diagonal.len(), diagonal.len(), &mut |r, c| if r == c {
			diagonal[r].clone()
		} else {
			R::zero()
		})
	}

	/// Creates an upper bidiagonal matrix from a diagonal and superdiagonal
	pub fn from_bidiagonal(diagonal: Vec<R>, superdiagonal: Vec<R>) -> Matrix<R> {
		debug_assert_eq!(diagonal.len(), superdiagonal.len() + 1);
		Matrix::from_index_def(diagonal.len(), diagonal.len(), &mut |r, c| if r == c {
			diagonal[r].clone()
		} else if c > 0 && r == c - 1 {
			superdiagonal[r].clone()
		} else {
			R::zero()
		})
	}

	/// Constructs a matrix defined index-wise
	pub fn from_index_def(
		rows: usize,
		cols: usize,
		at_index: &mut dyn FnMut(usize, usize) -> R) -> Matrix<R> {

		Matrix::from_flatmap(rows, cols, Vec::from_iter((0..(rows * cols)).map(|i|
			at_index(i % rows, i / rows)
		)))
	}

	/// Constructs a matrix with the given columns
	pub fn from_cols(columns: Vec<Matrix<R>>) -> Matrix<R> {
		let m = columns[0].row_count();
		// Make sure they are all vectors of the same size
		debug_assert!(columns.iter().map(|c| c.is_vector()).reduce(|acc, e| acc && e).unwrap());
		debug_assert!(columns.iter().map(|c| c.row_count() == m).reduce(|acc, e| acc && e).unwrap());

		let mut flatmap = vec![R::zero() ; columns.len() * m];

		for c in 0..columns.len() {
			for r in 0..m {
				flatmap[index!(m, columns.len(), r, c)] = columns[c].get(r, 0);
			}
		}

		Matrix { flatmap, row_count: m, col_count: columns.len() }
	}

	// MARK: Properties

	/// Returns the diagonal of this matrix as a list
	pub fn get_diagonal(&self) -> Vec<R> {
		let mut diagonal = vec![R::zero() ; min(self.row_count(), self.col_count())];
		for i in 0..diagonal.len() {
			diagonal[i] = self.get(i, i)
		}
		diagonal
	}

	/// Returns the upper diagonal of this matrix as a list
	pub fn get_upperdiagonal(&self) -> Vec<R> {
		let mut upper_diagonal = vec![R::zero() ; min(self.row_count(), self.col_count()) - 1];
		for i in 0..upper_diagonal.len() {
			upper_diagonal[i] = self.get(i, i + 1)
		}
		upper_diagonal
	}

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
	pub fn as_vec(&self) -> Vec<R> {
		self.flatmap.clone()
	}

	// MARK: Utility

	/// Returns the columns of this matrix
	pub fn columns(&self) -> Vec<Matrix<R>> {
		(0..self.col_count()).into_iter().map(|c| 
			Matrix::from_flatmap(self.row_count(), 1, self.flatmap[
				index!(self.row_count(), self.col_count(), 0, c)..index!(self.row_count(), self.col_count(), self.row_count(), c)
			].to_vec())
		).collect()
	}

	/// Returns true if this matrix is the identity matrix
	pub fn is_identity(&self) -> bool {
		for r in 0..self.row_count() {
			for c in 0..self.col_count() {
				if r == c {
					if self.get(r, c) != R::one() {
						return false;
					}
					else if self.get(r, c) != R::zero() {
						return false;
					}
				}
			}
		}

		return true;
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

	/// Appends a matrix to this matrix, on the bottom
	pub fn append_mat_bottom(&mut self, mat: Matrix<R>) {
		debug_assert_eq!(self.col_count(), mat.col_count());

		let original = self.clone();
		*self = Matrix::new(self.row_count() + mat.row_count(), self.col_count());

		for r in 0..original.row_count() {
			for c in 0..original.col_count() {
				self.set(r, c, original.get(r, c));
			}
		}

		for r in 0..mat.row_count() {
			for c in 0..mat.col_count() {
				self.set(original.row_count() + r, c, mat.get(r, c));
			}
		}
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

	/// Accesses a sub-matrix of this matrix
	pub fn get_submatrix(&self, row_range: Range<usize>, col_range: Range<usize>) -> Matrix<R> {
		let mut submat = Matrix::new(row_range.len(), col_range.len());

		for r in row_range.clone() {
			for c in col_range.clone() {
				submat.set(r - row_range.start, c - col_range.start, 
					self.get(r, c));
			}
		}

		submat
	}

	/// Writes to a sub-matrix of this matrix
	pub fn set_submatrix(&mut self, row_range: Range<usize>, col_range: Range<usize>, submat: Matrix<R>) {
		debug_assert_eq!(row_range.len(), submat.row_count());
		debug_assert_eq!(col_range.len(), submat.col_count());

		for r in row_range.clone() {
			for c in col_range.clone() {
				self.set(r, c,
					submat.get(r - row_range.start, c - col_range.start));
			}
		}
	}

	// MARK: Math

	/// Computes the inner-product of this vector with another vector
	/// 
	/// This operates in the entries in the flatmap of each matrix, so if 
	/// the argument to this function are not proper vectors (i.e., they 
	/// are matrices with both dimensions greater than 1) then the behavior
	/// here is not well-defined
	pub fn inner_product(&self, other: &Matrix<R>) -> R {
		debug_assert_eq!(self.flatmap.len(), other.flatmap.len());

		let mut inner_product = R::zero();

		for i in 0..self.flatmap.len() {
			inner_product += self.flatmap[i].clone() * other.flatmap[i].clone();
		}

		inner_product
	}

	/// The squared L2 norm of this vector
	pub fn l2_norm_squared(&self) -> R {
		self.inner_product(self)
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

impl<F: Field> Div<F> for Matrix<F> {
	type Output = Matrix<F>;

	fn div(self, rhs: F) -> Matrix<F> {
		let mut out = Matrix::<F>::new(self.row_count, self.col_count);
		scalar_div(self.row_count * self.col_count, 
			rhs, 
			&self.flatmap, &mut out.flatmap
		);
		out
	}
}

impl<F: Field> DivAssign<F> for Matrix<F> {
	fn div_assign(&mut self, rhs: F) {
		scalar_div_assign(self.row_count * self.col_count, 
			rhs, &mut self.flatmap
		);
	}
}

impl<R: Ring> Mul<Matrix<R>> for Matrix<R> {
	type Output = Matrix<R>;

	fn mul(self, rhs: Matrix<R>) -> Self::Output {
		assert_eq!(self.col_count, rhs.row_count, "Attempting to multiply a {} x {} matrix with a {} x {} matrix.", self.row_count(), self.col_count(), rhs.row_count(), rhs.col_count());

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

	#[test]
	fn test_transpose() {
		let a = Matrix::from_flatmap(16, 1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
		println!("{:?}", a.transpose());
	}

	#[test]
	fn test_index_def() {
		let a = Matrix::from_index_def(10, 9, &mut {|r, c|
			if r == c { 1 } else {0}
		} );

		println!("{:?}", a.transpose());
	}

}



// MARK: Matrices over Fields

impl<F: Field> Matrix<F> {

	/// Returns whether or not two vectors are orthogonal
	pub fn is_orthogonal_to(&self, other: Matrix<F>) -> bool {
		debug_assert!(compatible_vectors!(self, other));
		self.inner_product(&other) == F::zero()
	}

	/// Returns whether or not a matrix is orthogonal (meaning its columns 
	/// are each orthogonal and of unit length)
	pub fn is_orthogonal(&self) -> bool {
		(self.transpose() * self.clone()).is_identity()
	}

	/// Projects this vector onto another vector
	pub fn proj_onto(&self, other: Matrix<F>) -> Matrix<F> {
		debug_assert!(compatible_vectors!(self, other));
		let scalar = other.inner_product(self) / other.inner_product(&other);
		other * scalar
	}

	/// Performs Gram-Schmidt Orthogonalization on a matrix, returning a
	/// matrix whose columns are orthogonal, and span the same column space 
	/// as the original matrix. This does NOT normalize the GS vectors.
	pub fn gram_schmidt(&self) -> Matrix<F> {
		let v = self.columns();

		let mut u = v.clone();

		// The first GS vector is just the normalized regular guy
		u[0] = v[0].clone();

		for k in 1..u.len() {
				u[k] = v[k].clone();
			for i in 0..k {
				u[k] = u[k].clone() - u[k].proj_onto(u[i].clone())
			}
		}

		Matrix::from_cols(u)
	}

}

// MARK: Linear Algebra over the Reals

impl Matrix<f64> {

	// MARK: Matrix Initialization

	/// Constructs a random matrix with gaussianly distributed entries
    pub fn random_normal(rows: usize, cols: usize, mean: f64, variance: f64) -> Matrix<f64> {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(mean, variance).unwrap();
		Matrix::from_index_def(rows, cols, &mut |_, _| normal.sample(&mut rand_gen))
    }

	// MARK: Vector utility

	/// Normalizes this vector
	pub fn normalize(&mut self) {
		debug_assert!(self.is_vector());
		*self /= self.l2_norm_squared().sqrt();
	}

	/// Returns the normalized version of this vector
	pub fn normalized(&self) -> Matrix<f64> {
		debug_assert!(self.is_vector());
		let mut unit = self.clone();
		unit.normalize();
		unit
	}

	/// Returns the angle of this 2D vector with the x-axis
	pub fn angle(&self) -> f64 {
		debug_assert!(self.is_vector());
		debug_assert_eq!(self.row_count(), 2);

		let x = self.get(0, 0);
		let y = self.get(1, 0);

		y.atan2(x)
	}

}