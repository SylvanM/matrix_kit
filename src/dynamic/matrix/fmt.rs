use std::fmt::Debug;
use crate::index;
use super::Matrix;

// MARK: Debug

impl<T: Debug + Clone> Debug for Matrix<T> {
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

#[cfg(test)]
mod tests {
	use super::*;

}
