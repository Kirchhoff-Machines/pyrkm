"""
Weight Suppression Toolkit for RKM Models
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import torch


class WeightSuppressionAnalyzer:
    """
    Simplified analyzer for iterative node removal in RKM models.

    This version only supports the iterative suppression method used in the analysis notebook.
    """

    def __init__(self, model, test_data: torch.Tensor):
        """
        Initialize the analyzer with a trained RKM model and test data.

        Parameters
        ----------
        model : RKM
            A trained RKM model
        test_data : torch.Tensor
            Test dataset for evaluation
        """
        self.original_model = model
        self.test_data = test_data.to(model.device).to(model.mytype)
        self.device = model.device
        self.mytype = model.mytype

        # Store baseline performance
        self.baseline_metrics = self._compute_baseline_metrics()

    def _compute_baseline_metrics(self) -> Dict:
        """Compute baseline metrics for the original model."""
        with torch.no_grad():
            reconstructed = self.original_model.forward(
                self.test_data, self.original_model.k)
            reconstruction_error = torch.mean(
                (self.test_data - reconstructed)**2).item()

        return {
            'reconstruction_error': reconstruction_error,
            'n_weights': self.original_model.W.numel(),
            'n_hidden_nodes': self.original_model.n_hidden
        }

    def iterative_suppression_with_fine_tuning(self,
                                               target_sparsity: float = 0.5,
                                               fine_tune_epochs: int = 5,
                                               lr_factor: float = 0.1,
                                               train_data: Optional[
                                                   torch.Tensor] = None,
                                               verbose: bool = True) -> Dict:
        """
        Iteratively suppress the least used hidden nodes with fine-tuning after each removal.

        This method implements a gradual suppression approach:
        1. Identify the least used hidden node (lowest total connection strength)
        2. Suppress the entire node (set all its connections to zero)
        3. Fine-tune the remaining weights to accommodate the new structure
        4. Repeat until target sparsity is reached

        Parameters
        ----------
        target_sparsity : float, default=0.5
            Target sparsity ratio (0.0 to 1.0). E.g., 0.5 means 50% of weights suppressed
        max_iterations : int, default=50
            Maximum number of suppression iterations
        fine_tune_epochs : int, default=5
            Number of epochs for fine-tuning after each suppression
        lr_factor : float, default=0.1
            Learning rate reduction factor for fine-tuning
        train_data : torch.Tensor, optional
            Training data for fine-tuning. If None, uses test data (not recommended)
        performance_threshold : float, default=1.5
            Stop if reconstruction error ratio exceeds this threshold
        verbose : bool, default=True
            Whether to print progress information during suppression

        Returns
        -------
        Dict
            Results tracking the iterative suppression process
        """
        if verbose:
            print(
                f"🔄 Starting iterative suppression targeting {target_sparsity:.1%} sparsity..."
            )

        if train_data is None and verbose:
            print(
                '⚠️ Warning: No training data provided, using test data for fine-tuning (not recommended)'
            )
            train_data = self.test_data

        # Initialize working model and results
        current_model = copy.deepcopy(self.original_model)
        results = {}
        iteration = 0

        # Calculate baseline metrics
        baseline_reconstruction_error = self.baseline_metrics[
            'reconstruction_error']
        baseline_fid = 1.0  # Simplified FID score

        # Track progress
        total_weights = current_model.W.numel() + current_model.h_bias.numel(
        ) + current_model.v_bias.numel()
        target_suppressed_weights = int(target_sparsity * total_weights)

        if verbose:
            print(f"Target: suppress {target_suppressed_weights} weights "
                  f"({target_sparsity:.1%} of {total_weights} total)")

        # Calculate how many nodes have been removed so far (0 expected at start)
        removed_nodes = (current_model.W.abs().sum(dim=1) == 0).sum().item()
        target_nodes_to_remove = int(target_sparsity * current_model.n_hidden)
        if removed_nodes > 0:
            print(
                f"⚠️ Warning: Model already has {removed_nodes} removed nodes at start"
            )
        list_of_removed_nodes = set()

        iteration = removed_nodes
        while iteration < target_nodes_to_remove:

            # Find least used hidden node
            with torch.no_grad():
                # Calculate connection strength for each hidden node (notice that I am ignoring biases here)
                weight_strength = torch.sum(torch.abs(current_model.W),
                                            dim=1)  # [n_hidden]

                # Find the id of the least used active node that is not already removed
                active_nodes = (weight_strength
                                != 0) & (~torch.tensor([
                                    i in list_of_removed_nodes
                                    for i in range(current_model.n_hidden)
                                ],
                                                       device=self.device))
                if active_nodes.sum() == 0:
                    if verbose:
                        print('❌ No active nodes remaining')
                    break

                # FIX: Get the actual node index from the masked tensor
                active_indices = torch.where(active_nodes)[0]
                least_strength, masked_idx = torch.min(
                    weight_strength[active_nodes], dim=0)
                least_used_idx = active_indices[masked_idx]
                list_of_removed_nodes.add(least_used_idx.item())

                # VERIFICATION: Check that exactly 'iteration+1' nodes are masked
                currently_masked = (torch.sum(torch.abs(current_model.W),
                                              dim=1) == 0).sum().item()
                if currently_masked != iteration:
                    if verbose:
                        print(
                            f"⚠️ Warning: Expected {iteration} masked nodes, found {currently_masked}"
                        )

                if verbose:
                    print(
                        f"Suppressing node {least_used_idx.item()} (strength: {least_strength:.4f})"
                    )

                if verbose:
                    print(
                        f"Suppressing node {least_used_idx} (strength: {least_strength:.4f})"
                    )

            # Suppress the least used hidden node completely
            with torch.no_grad():
                current_model.W[
                    least_used_idx, :] = 0.0  # All visible connections to this hidden node
                current_model.h_bias[least_used_idx] = 0.0  # Hidden bias
                # Update transpose matrix immediately
                current_model.W_t = current_model.W.t()

                # VERIFICATION: Confirm exactly iteration+1 nodes are now masked
                masked_nodes = (torch.sum(torch.abs(current_model.W),
                                          dim=1) == 0).sum().item()
                expected_masked = iteration + 1
                if masked_nodes != expected_masked:
                    if verbose:
                        print(
                            f"❌ Error: Expected {expected_masked} masked nodes after suppression,"
                            f" found {masked_nodes}")
                elif verbose:
                    print(
                        f"  ✅ Verified: {masked_nodes} nodes correctly masked")

            # Create suppression mask for fine-tuning
            suppression_mask = (current_model.W == 0)

            # Fine-tune the model with the new structure
            if fine_tune_epochs > 0:
                if verbose:
                    print(f"  Fine-tuning for {fine_tune_epochs} epochs...")
                try:
                    current_model = self.fine_tune_suppressed_model(
                        current_model,
                        suppression_mask,
                        train_data,
                        n_epochs=fine_tune_epochs,
                        lr_factor=lr_factor,
                        verbose=verbose)

                    # CRITICAL: Ensure suppressed weights remain zero after fine-tuning
                    with torch.no_grad():
                        current_model.W[least_used_idx, :] = 0.0
                        current_model.h_bias[least_used_idx] = 0.0
                        current_model.W_t = current_model.W.t()

                except Exception as e:
                    if verbose:
                        print(f"  Fine-tuning failed: {e}")

            # Evaluate current performance
            with torch.no_grad():
                reconstructed = current_model.forward(self.test_data,
                                                      current_model.k)
                current_reconstruction_error = torch.mean(
                    (self.test_data - reconstructed)**2).item()
                current_fid = 1  # Simplified for now

            reconstruction_error_ratio = current_reconstruction_error / baseline_reconstruction_error
            fid_ratio = current_fid / baseline_fid

            # Store iteration results
            iteration_name = f'iteration_{iteration+1:02d}'

            results[iteration_name] = {
                'model': copy.deepcopy(current_model),
                'iteration': iteration + 1,
                'suppressed_node': least_used_idx.item(),
                'node_strength': least_strength,
                'reconstruction_error': current_reconstruction_error,
                'reconstruction_error_ratio': reconstruction_error_ratio,
                'fid_score': current_fid,
                'fid_ratio': fid_ratio,
                'total_weights': total_weights
            }

            # Show progress every 5 iterations or at key milestones
            if verbose and (iteration % 5 == 0
                            or reconstruction_error_ratio > 1.5):
                print(
                    f"  Iter {iteration+1}: Node remaining {current_model.n_hidden-len(removed_nodes) :.1%}"
                    f", Performance ratio {reconstruction_error_ratio:.3f}")

            iteration += 1

        return results

    def fine_tune_suppressed_model(self,
                                   model,
                                   suppression_mask: torch.Tensor,
                                   train_data: torch.Tensor,
                                   n_epochs: int = 10,
                                   lr_factor: float = 0.1,
                                   verbose: bool = True):
        """
        Fine-tune a suppressed model using RKM's native masking system.

        Parameters
        ----------
        model : RKM
            Model with suppressed weights
        suppression_mask : torch.Tensor
            Boolean mask indicating which weights are suppressed
        train_data : torch.Tensor
            Training data for fine-tuning
        n_epochs : int, default=10
            Number of fine-tuning epochs
        lr_factor : float, default=0.1
            Factor to reduce learning rate for fine-tuning
        verbose : bool, default=True
            Whether to print progress information

        Returns
        -------
        torch.nn.Module
            Fine-tuned model
        """

        # Store original parameters for restoration
        original_lr = model.lr
        original_max_epochs = model.max_epochs
        original_W_mask = model.W_mask
        original_v_bias_mask = model.v_bias_mask
        original_h_bias_mask = model.h_bias_mask

        # Set fine-tuning parameters
        model.lr = original_lr * lr_factor
        model.max_epochs = n_epochs

        # The RKM will automatically prevent updates to masked weights during training
        model.W_mask = suppression_mask.to(model.device)

        # For completely suppressed nodes, we also need to mask the biases
        # Find which hidden nodes are completely suppressed (all weights = 0)
        suppressed_hidden_nodes = torch.all(suppression_mask,
                                            dim=1)  # [n_hidden] boolean
        model.h_bias_mask = suppressed_hidden_nodes.to(model.device)

        # We don't suppress visible biases in this method, but set mask to None for clarity
        model.v_bias_mask = None

        # Ensure suppressed weights are zero before training
        with torch.no_grad():
            model.W[model.W_mask] = 0.0
            model.h_bias[model.h_bias_mask] = 0.0
            model.W_t = model.W.t()

        # Create data loader
        train_loader = torch.utils.data.DataLoader(train_data.to(model.device),
                                                   batch_size=model.batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

        # Use a subset of test data for validation during fine-tuning
        test_data_dummy = train_data[:min(100, len(train_data))].to(
            model.device)

        try:
            # Use RKM's native training - the masking will be automatically applied
            # in SGD_update() and Adam_update() methods
            model.train(train_loader, test_data_dummy, print_error=False)

        finally:
            # Restore original parameters
            model.lr = original_lr
            model.max_epochs = original_max_epochs
            model.W_mask = original_W_mask
            model.v_bias_mask = original_v_bias_mask
            model.h_bias_mask = original_h_bias_mask

        if verbose:
            print("Fine-tuning completed using RKM's native masking system!")

        return model
