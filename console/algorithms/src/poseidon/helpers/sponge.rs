// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use crate::poseidon::{helpers::AlgebraicSponge, State};
use snarkvm_console_types::{prelude::*, Field};
use snarkvm_fields::PoseidonParameters;

use smallvec::SmallVec;
use std::sync::Arc;

/// A duplex sponge based using the Poseidon permutation.
///
/// This implementation of Poseidon is entirely from Fractal's implementation in [COS20][cos]
/// with small syntax changes.
///
/// [cos]: https://eprint.iacr.org/2019/1076
#[derive(Clone, Debug)]
pub struct PoseidonSponge<E: Environment, const RATE: usize, const CAPACITY: usize> {
    /// Sponge Parameters
    parameters: Arc<PoseidonParameters<E::Field, RATE, CAPACITY>>,
    /// Current sponge's state (current elements in the permutation block)
    state: State<E, RATE, CAPACITY>,
}

impl<E: Environment, const RATE: usize, const CAPACITY: usize> AlgebraicSponge<E, RATE, CAPACITY>
    for PoseidonSponge<E, RATE, CAPACITY>
{
    type Parameters = Arc<PoseidonParameters<E::Field, RATE, CAPACITY>>;

    fn new(parameters: &Self::Parameters) -> Self {
        Self { parameters: parameters.clone(), state: State::default() }
    }

    fn absorb(&mut self, input: &[Field<E>]) {
        if !input.is_empty() {
            let last_chunk_index = input.len() / RATE;
            for (i, chunk) in input.chunks(RATE).enumerate() {
                for (element, state_elem) in chunk.iter().zip(self.state.rate_state_mut()) {
                    *state_elem += element;
                }
                // Still chunks ahead? If so, let's permute
                if i < last_chunk_index {
                    self.permute();
                }
            }
        }
    }

    fn squeeze(&mut self, num_elements: u16) -> SmallVec<[Field<E>; 10]> {
        if num_elements == 0 {
            return SmallVec::new();
        }
        let mut output = if num_elements <= 10 {
            smallvec::smallvec_inline![Field::<E>::zero(); 10]
        } else {
            smallvec::smallvec![Field::<E>::zero(); num_elements as usize]
        };

        self.permute();
        self.squeeze_internal(&mut output[..num_elements as usize]);

        output.truncate(num_elements as usize);
        output
    }
}

impl<E: Environment, const RATE: usize, const CAPACITY: usize> PoseidonSponge<E, RATE, CAPACITY> {
    #[inline]
    fn apply_ark(&mut self, round_number: usize) {
        for (state_elem, ark_elem) in self.state.iter_mut().zip(&self.parameters.ark[round_number]) {
            *state_elem += Field::<E>::new(*ark_elem);
        }
    }

    #[inline]
    fn apply_s_box(&mut self, is_full_round: bool) {
        // Full rounds apply the S Box (x^alpha) to every element of state
        let alpha = Field::from_u64(self.parameters.alpha);
        if is_full_round {
            for elem in self.state.iter_mut() {
                *elem = elem.pow(alpha);
            }
        }
        // Partial rounds apply the S Box (x^alpha) to just the first element of state
        else {
            self.state[0] = self.state[0].pow(alpha);
        }
    }

    #[inline]
    fn apply_mds(&mut self) {
        let mut new_state = State::default();
        new_state.iter_mut().zip(&self.parameters.mds).for_each(|(new_elem, mds_row)| {
            *new_elem = Field::new(E::Field::sum_of_products(self.state.iter().map(|e| e.deref()), mds_row.iter()));
        });
        self.state = new_state;
    }

    #[inline]
    fn permute(&mut self) {
        // Determine the partial rounds range bound.
        let partial_rounds = self.parameters.partial_rounds;
        let full_rounds = self.parameters.full_rounds;
        let full_rounds_over_2 = full_rounds / 2;
        let partial_round_range = full_rounds_over_2..(full_rounds_over_2 + partial_rounds);

        // Iterate through all rounds to permute.
        for i in 0..(partial_rounds + full_rounds) {
            let is_full_round = !partial_round_range.contains(&i);
            self.apply_ark(i);
            self.apply_s_box(is_full_round);
            self.apply_mds();
        }
    }

    /// Squeeze |output| many elements. This does not end in a squeeze
    #[inline(always)]
    fn squeeze_internal(&mut self, output: &mut [Field<E>]) {
        // The total number of chunks is `output.len() / RATE`, plus 1 for the remainder.
        let output_len = output.len();
        let last_chunk_index = output_len / RATE;

        // Absorb the input output, `RATE` output at a time, except for the first chunk, which
        // is of size `RATE - rate_start`.
        for (i, chunk) in output.chunks_mut(RATE).enumerate() {
            let range = 0..chunk.len();
            debug_assert_eq!(
                chunk.len(),
                self.state.rate_state(range.clone()).len(),
                "Failed to squeeze {} at rate {RATE}",
                output_len
            );
            chunk.copy_from_slice(self.state.rate_state(range));
            // Still chunks ahead? If so, let's permute
            if i < last_chunk_index {
                self.permute();
            }
        }
    }
}
