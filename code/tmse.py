import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

    """
    Reference: 
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

class _GaussianSimilarityTMSE(nn.Module):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(
        self, threshold: float = 1, sigma: float = 1, ignore_index: int = 255
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        batch 내에 있는 x_t, x_(t+1)의 값을 gaussian weighted로 하는 segmentation loss 
        
        Args:
            preds: the output of model before softmax. (N, C, T) [batch, feature, temporal]
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)  - feature [batch, feature, temporal]
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]
        print(preds.shape)
        print(gts.shape)
        print(sim_index.shape)
        for i, (pred, gt, sim) in enumerate(zip(preds, gts, sim_index)):

            # calculate gaussian similarity
            if i != batch_size - 1 :
                diff = sim - sim_index[i+1]
                similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * self.sigma ** 2))
                # calculate temporal mse
                loss = self.mse(
                                F.log_softmax(pred), F.log_softmax(preds[i+1, :])
                            )
                loss = torch.clamp(loss, min=0, max=self.threshold ** 2)

                # gaussian similarity weighting
                loss = similarity * loss

                total_loss += torch.mean(loss)

        return total_loss



    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(
        self,
        bce: bool = True,
        mse: bool = False,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.criterions = []

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
            )

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                loss += criterion(pred[mask], gt[mask])

        return loss / batch_size