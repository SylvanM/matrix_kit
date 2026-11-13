use std::ops::Range;
use algebra_kit::algebra::Ring;
use crate::index;
use super::Matrix;

// MARK: Shape & Layout

impl<T: Clone> Matrix<T> {

	/// Returns the columns of this matrix
	pub fn columns(&self) -> Vec<Matrix<T>> {
		(0..self.col_count()).into_iter().map(|c| 
			Matrix::from_flatmap(self.row_count(), 1, self.flatmap[
				index!(self.row_count(), self.col_count(), 0, c)..index!(self.row_count(), self.col_count(), self.row_count(), c)
			].to_vec())
		).collect()
	}

	/// Appends a reference to a column to this matrix on the right
	pub fn append_col_ref(&mut self, col: &mut Vec<T>) { 
		assert_eq!(col.len(), self.row_count());

		self.col_count += 1;
		self.flatmap.append(col);
	}

	/// Appends a reference to a matrix to this matrix on the right
	pub fn append_mat_right_ref(&mut self, mat: &mut Matrix<T>) {
		assert_eq!(self.row_count(), mat.row_count());

		self.col_count += mat.col_count();
		self.flatmap.append(&mut mat.flatmap);
	}

	/// Appends a column to this matrix on the right
	pub fn append_col(&mut self, col: Vec<T>) {
		assert_eq!(col.len(), self.row_count());

		let mut other_col = col.clone();
		self.flatmap.append(&mut other_col);

		self.col_count += 1;
	}

	/// Appends a matrix to this matrix, on the right
	pub fn append_mat_right(&mut self, mat: Matrix<T>) {
		assert_eq!(self.row_count(), mat.row_count());
		assert!(!mat.flatmap.is_empty());
		let mut new_flatmap = vec![mat.get(0, 0) ; self.flatmap.len() + mat.flatmap.len()];

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
	pub fn append_row(&mut self, row: Vec<T>) {
		assert!(!row.is_empty());
		assert_eq!(row.len(), self.col_count());

		let mut new_flatmap = vec![row[0].clone() ; self.flatmap.len() + row.len()];
		
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
	pub fn append_mat_bottom(&mut self, mat: Matrix<T>) {
		debug_assert_eq!(self.col_count(), mat.col_count());


		let new = Matrix::from_index_def(self.row_count() + mat.row_count(), self.col_count(), &mut |r, c|
			if r < self.row_count() {
				self.get(r, c)
			} else {
				mat.get(r - self.row_count(), c)
			}
		);

		*self = new
	}

	// MARK: Utility

	/// Applies a function to all entries in this matrix, returning the result 
	/// as a separate matrix
	pub fn applying_to_all<J: Ring>(&self, f: &dyn Fn(T) -> J) -> Matrix<J> {
		Matrix { 
			flatmap: self.flatmap.iter().map(|x| f(x.clone())).collect(), 
			row_count: self.row_count(), 
			col_count: self.col_count() 
		}
	}

	/// Applies a function to all entries in this matrix, in place
	pub fn apply_to_all(&mut self, f: &dyn Fn(T) -> T) {
		for i in 0..self.flatmap.len() {
			self.flatmap[i] = f(self.flatmap[i].clone())
		}
	}

	/// The transpose of this matrix 
	pub fn transpose(&self) -> Matrix<T> {
		Matrix::from_index_def(self.col_count(), self.row_count, &mut |r, c| self.get(c, r))
	}

	/// Accesses a sub-matrix of this matrix
	pub fn get_submatrix(&self, row_range: Range<usize>, col_range: Range<usize>) -> Matrix<T> {
		Matrix::from_index_def(row_range.len(), col_range.len(), &mut |r, c| 
			self.get(r + row_range.start, c + col_range.start)
		)
	}

	/// Writes to a sub-matrix of this matrix
	pub fn set_submatrix(&mut self, row_range: Range<usize>, col_range: Range<usize>, submat: Matrix<T>) {
		debug_assert_eq!(row_range.len(), submat.row_count());
		debug_assert_eq!(col_range.len(), submat.col_count());

		debug_assert!(row_range.end <= self.row_count());
		debug_assert!(col_range.end <= self.col_count());

		for r in row_range.clone() {
			for c in col_range.clone() {
				self.set(r, c,
					submat.get(r - row_range.start, c - col_range.start));
			}
		}
	}

	/// Swaps two rows of a matrix
	pub fn swap_rows(&mut self, r1: usize, r2: usize) {
		if r1 == r2 { return; }
		let m = self.row_count();
		for col in self.flatmap.chunks_exact_mut(m) {
			col.swap(r1, r2);
		}
	}


	/// Swaps two columns of a matrix
	pub fn swap_cols(&mut self, c1: usize, c2: usize) {
		if c1 == c2 { return; }
		let m = self.row_count();
		let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };

		let (left, right) = self.flatmap.split_at_mut(hi * m);
		left[lo * m..(lo + 1) * m].swap_with_slice(&mut right[..m]);
	}


}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_transpose() {
		let a = Matrix::from_flatmap(16, 1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
		println!("{:?}", a.transpose());
	}

}
