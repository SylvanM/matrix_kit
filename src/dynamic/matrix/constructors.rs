use std::cmp::min;
use algebra_kit::algebra::Ring;
use rand_distr::Distribution;
use crate::index;
use super::Matrix;

impl<R: Ring> Matrix<R> {

	// MARK: Initializers

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

	/// Constructs a matrix of all zeroes for a given dimension
	pub fn new(rows: usize, cols: usize) -> Matrix<R> {
		Matrix::from_flatmap(rows, cols, vec![R::zero() ; rows * cols])
	}

	/// Constructs a matrix of all ones
	pub fn ones(rows: usize, cols: usize) -> Matrix<R> {
		Matrix { 
			flatmap: vec![R::one() ; rows * cols], 
			row_count: rows, 
			col_count: cols 
		}
	}

	/// Returns the diagonal of this matrix as a list
	pub fn get_diagonal(&self) -> Vec<R> {
		let mut diagonal = vec![R::zero() ; min(self.col_count(), self.row_count())];
		for i in 0..diagonal.len() {
			diagonal[i] = self.get(i, i)
		}
		diagonal
	}

	/// Returns the upper diagonal of this matrix as a list
	pub fn get_upperdiagonal(&self) -> Vec<R> {
		let mut upper_diagonal = vec![R::zero() ; min(self.col_count(), self.row_count()) - 1];
		for i in 0..upper_diagonal.len() {
			upper_diagonal[i] = self.get(i, i + 1)
		}
		upper_diagonal
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

	/// Creates a block-diagonal matrix from square blocks
	pub fn from_block_diagonal(blocks: Vec<Matrix<R>>) -> Matrix<R> {
		debug_assert!(blocks.iter().map(|b| b.is_square()).reduce(|b1, b2| b1 && b2).unwrap());

		let sizes: Vec<usize> = blocks.iter().map(|b| b.col_count()).collect();
		let n = sizes.iter().sum();

		let mut matrix = Matrix::new(n, n);

		let mut current_start = 0;

		for i in 0..blocks.len() {
			matrix.set_submatrix(
				current_start..(current_start + sizes[i]), 
				current_start..(current_start + sizes[i]), 
				blocks[i].clone()
			);
			current_start += sizes[i];
		}

		matrix
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
}

impl Matrix<f64> {

	/// Constructs a random matrix with gaussianly distributed entries
    pub fn random_normal(rows: usize, cols: usize, mean: f64, variance: f64) -> Matrix<f64> {
        let mut rand_gen = rand::rng();
        let normal = rand_distr::Normal::new(mean, variance).unwrap();
		Matrix::from_index_def(rows, cols, &mut |_, _| normal.sample(&mut rand_gen))
    }

}

#[cfg(test)]
mod tests {
	use super::*;

}
