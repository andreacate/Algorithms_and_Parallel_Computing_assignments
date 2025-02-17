#include "simulator.hpp"

#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <mpi.h>

Simulator::Simulator(const la::dense_matrix& proposer, const la::dense_matrix& acceptor)
    : preferences_proposer(proposer), preferences_acceptor(acceptor), num_elements(proposer.rows()) {
  matches         = la::dense_matrix(num_elements, 1, num_elements); // we start without matches
  proposer_status = la::dense_matrix(num_elements, 1, 0);            // we start with the best choice
}

la::dense_matrix::value_type Simulator::select_best_proposer(const value_type acceptor_index,
                                                             const value_type candidate_1,
                                                             const value_type candidate_2) const {
  for (value_type candidate_index = 0; candidate_index < num_elements; ++candidate_index) {
    const value_type proposer_index = preferences_acceptor(acceptor_index, candidate_index);
    if (proposer_index == candidate_1) {
      return candidate_1;
    } else if (proposer_index == candidate_2) {
      return candidate_2;
    }
  }
  return candidate_1; // both candidates are unknown, we pick the first one
}

la::dense_matrix::value_type Simulator::get_matching_proposer(const value_type acceptor_index) const {
  return matches(acceptor_index, 0);
}

la::dense_matrix::value_type Simulator::get_matching_acceptor(const value_type proposer_index) const {
  for (value_type acceptor_index = 0; acceptor_index < matches.rows(); ++acceptor_index) {
    if (matches(acceptor_index, 0) == proposer_index) {
      return acceptor_index;
    }
  }
  return matches.rows();
}

void Simulator::update_matches(const la::dense_matrix& proposal) {
  // propasal is a num_element x num_elements matrix. Each row represents an acceptor.
  // Each column represents a proposer. All the elements can assume value zero or one.
  // The elements with value one means that the proposer has made a proposal to the
  // related acceptor.
  // Matches is a num_element x 1 matrix. Where each row represents an acceptor. The
  // value of the column is the index of the matching proposer. If there is no match, we
  // use the num_element value.
  // You can split the work by assigning different rows to different process and combine
  // togheter the result.

  // YOUR CODE GOES HERE
  int size=0, rank=0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  la::dense_matrix local_matches(num_elements,1,0);

  // loop over all the acceptors
  for (value_type acceptor_index = rank; acceptor_index < num_elements; acceptor_index+=size) {
    // find out which is the best proposer that we have (considering the current matching)
    const value_type previous_match = get_matching_proposer(acceptor_index);
    value_type best_proposer_index  = previous_match;
    for (value_type proposer_index = 0; proposer_index < num_elements; ++proposer_index) {
      const bool is_a_match = proposal(acceptor_index, proposer_index) == 1;
      if (is_a_match) {
        best_proposer_index = select_best_proposer(acceptor_index, best_proposer_index, proposer_index);
      }
    }

    // store the best option in the new matching matrix
    local_matches(acceptor_index, 0) = best_proposer_index;
    log_match(acceptor_index, best_proposer_index, previous_match);
  }

  MPI_Allreduce(local_matches.data(),matches.data(),num_elements, MPI_UNSIGNED,MPI_SUM, MPI_COMM_WORLD);

  // YOUR CODE GOES HERE
  return ;
}

la::dense_matrix Simulator::compute_proposal() {
  // the output of this method is a num_element x num_element matrix, where each row represents an
  // acceptor and each column represents a proposer. The matrix is zero initialized. Each proposer
  // that has not been paired with with an acceptor, it will propose to the most preferred one that
  // didn't reject it yet by:
  //  1 - writing 1 to the corresponging acceptor/proposer cell inside the proposal matrix
  //  2 - updating the proposer_status, increasing the index of the next best acceptor by one
  // You can split computation by assigning different rows to different processes. Then you must
  // combine the updates on the same data structures

  // declare the matrix that will hold the proposal
  la::dense_matrix proposal(num_elements, num_elements, 0);

  // YOUR CODE GOES HERE
  int size=0, rank=0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  la::dense_matrix local_proposer_status(num_elements, 1, 0);
  

  // loop over the proposer to fille the proposal matrix
  for (value_type proposer_index = rank; proposer_index < num_elements; proposer_index+=size) {
    local_proposer_status(proposer_index,0)=proposer_status(proposer_index,0);

    // get the current match for the current proposer
    const value_type current_match_index = get_matching_acceptor(proposer_index);

    // if the proposer doesn't have a match it needs to propose to the next best choice. Otherwise
    // the current match is the best that it can achieve
    if (current_match_index == num_elements) {
      // update the data structure that represents the proposal
      const value_type next_best_index =
          preferences_proposer(proposer_index, proposer_status(proposer_index, 0));
      proposal(next_best_index, proposer_index) = 1;

      // update the data structure that keep tracks of the most preffered choices for the porposers that do not
      // have rejected the proposer yet
      local_proposer_status(proposer_index, 0) = std::min(num_elements - 1, proposer_status(proposer_index, 0) + 1);

      log_proposal(proposer_index, next_best_index);
    } else {
      log_no_proposal(proposer_index, current_match_index);
    }
  }

  // YOUR CODE GOES HERE
   MPI_Allreduce (MPI_IN_PLACE, proposal.data(), num_elements*num_elements, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

   MPI_Allreduce (local_proposer_status.data(), proposer_status.data(), num_elements, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

  return proposal;
}

la::dense_matrix Simulator::run() {
  bool is_stable = false;
  while (!is_stable) {
    // compute the next round of the match making
    const la::dense_matrix new_proposal = compute_proposal();
    update_matches(new_proposal);

    // if all the proposers have been matched, we found a stable match
    bool current_is_stable = true;
    for (value_type proposer_index = 0; proposer_index < matches.rows() && current_is_stable;
         ++proposer_index) {
      const value_type match_index = get_matching_acceptor(proposer_index);
      if (match_index == num_elements) {
        current_is_stable = false;
      }
    }

    // check if we need to replace the new matching with the previous one
    if (current_is_stable) {
      is_stable = true;
    }
  }
  return matches;
}
