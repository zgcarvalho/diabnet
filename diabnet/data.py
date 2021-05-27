import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Any, Tuple

__all__ = ["DiabDataset", "get_feature_names", "encode_features"]


def get_feature_names(
    fn: str,
    use_sex: bool = True,
    use_parents_diagnosis: bool = True,
) -> List[str]:
    """Get feature names from a CSV file with feature names as headers.
    The columns containing IDs, labels and unnamed columns are ignored.

    Parameters
    ----------
    fn : str
        Path to CSV file with data. The first line must have feature names.
    use_sex : bool
        Whether to use `sex` as a feature, by default True.
    use_parents_diagnosis : bool
        Whether to use parents' diagnosis as feature, by default True.

    Returns:
    List[str]
        A list with feature names that will be used by DiabNet.

    Raises
    ------
    TypeError
        `fn` must be a string.
    ValueError
        `fn` must be a CSV-formatted file.
    TypeError
        `use_sex` must be a boolean.
    TypeError
        `use_parents_diagnosis` must be a boolean.
    """
    # Check arguments
    if type(fn) not in [str]:
        raise TypeError("`fn` must be a string.")
    if not fn.endswith(".csv"):
        raise ValueError("`fn` must be a CSV-formatted file.")
    if type(use_sex) not in [bool]:
        raise TypeError("`use_sex` must be a boolean.")
    if type(use_parents_diagnosis) not in [bool]:
        raise TypeError("`use_parents_diagnosis` must be a boolean.")

    # Read all headers from file
    all_headers = pd.read_csv(fn).columns.values

    # always remove headers:
    # ["id", "famid", "T2D(label)", "mo", "fa"]
    # Note: "mo_t2d" and "fa_t2d" header are also removed here, but they are
    # inserted into the correct position later.
    headers = np.setdiff1d(
        all_headers,
        np.array(
            [
                "id",
                "famid",
                "T2D",
                "mo",
                "fa",
                "mo_t2d",
                "fa_t2d",
                "BMI",
                "sex",
                "Unnamed: 0",
                "",
                " ",
            ]
        ),
        assume_unique=True,
    )

    # Convert headers to list
    features: List[str] = list(headers)

    # Add sex as a feature
    if use_sex:
        features.append("sex")

    # Add use parents diagnosis as a feature
    if use_parents_diagnosis:
        features.append("mo_t2d")
        features.append("fa_t2d")

    return features


def encode_features(feat_names: List[str], feat_values: List[Any]) -> np.ndarray:
    """Encode features into a numpy.ndarray to be used as input to DiabNet.

    Parameters
    ----------
    feat_names : List[str]
        A list of feature names.
    feat_values : List[Any]
        A list of feature values.

    Returns
    -------
    np.ndarray
        A numpy.ndarray with shape (2, len_encoded_features) for feature
        values to be used as input to DiabNet.

    Raises
    ------
    ValueError
        Unknown feature: `{feat_name}`.
    """
    # Check if parents diagnosis is in the correct position
    # FIXME: Check redundancy - use in other functions
    _check_parents_diag(feat_names)

    # Create a zeroed numpy.ndarray for all valid features
    m = np.zeros((2, _len_encoding(feat_names)))

    # Fill numpy.ndarray with encoded feature values
    for i, feat_name in enumerate(feat_names):
        # NOTE: All encoded feature values must be in the range ~[0, 2]

        # AGE
        if feat_name == "AGE":
            # Encoding: divide to 50 to put AGE in the range ~[0,2]
            m[0, i] = feat_values[i] / 50
            # Works like a bias in a layer without bias
            m[1, i] = 1

        # SEX
        elif feat_name == "sex":
            # Enconding: keep sex in the range [0,2]
            if feat_values[i] == "F":
                m[0, i] = 2
            elif feat_values[i] == "M":
                m[1, i] = 2

        # Mother T2D diagnosis
        elif feat_name == "mo_t2d":
            # If parent diagnosis are included in features,
            #  both 'mo_t2d' and 'fa_t2d' must be present.
            # The sequence must be ['mo_t2d', 'fa_t2d'] in
            #  the two last positions
            # Encoding
            if feat_values[i] == 0:
                m[0, i] = 1  # keep feat in the range [0,2]
            elif feat_values[i] == 1:
                m[0, i + 1] = 1
            else:
                m[0, i + 2] = 1
            # Works like a bias in a layer without bias
            m[1, i], m[1, i + 1], m[1, i + 2] = 1, 1, 1

        # Father T2D diagnosis
        elif feat_name == "fa_t2d":
            # If parent diagnosis are included in features,
            #  both 'mo_t2d' and 'fa_t2d' must be present.
            # The sequence must be ['mo_t2d', 'fa_t2d'] in
            #  the two last positions
            # Encoding
            if feat_values[i] == 0:
                m[0, i + 2] = 1
            elif feat_values[i] == 1:
                m[0, i + 3] = 1
            else:
                m[0, i + 4] = 1
            # Works like a bias in a layer without bias
            m[1, i + 2], m[1, i + 3], m[1, i + 4] = 1, 1, 1

        # SNPs
        elif feat_name[:3] == "snp":
            # Dominant homozygous
            if feat_values[i] == 0:
                m[0, i] = 2
            # Recessive homozygous
            elif feat_values[i] == 2:
                m[1, i] = 2
            # Heterozygous
            else:
                m[0, i], m[1, i] = 1, 1

        # Unknown feature -> Raise ValueError
        else:
            # if feature name is invalid raise an exception
            raise ValueError(f"Unknown feature: `{feat_name}`.")

    return m


def _check_parents_diag(feat_names: List[str]) -> None:
    """Checks if parents' diagnosis are in the correct position.
    The 'mo_t2d' and 'fa_t2d' MUST be the two last position in this
    specific order.

    Parameters
    ----------
    feat_names : List[str]
        A list of feature names.

    Raises
    ------
    ValueError
        mo_t2d is at index `x` and fa_t2d is at index `y` from
        feat_names with `n` items. They must be in the penultimate
        and last positions, respectively.
    """
    # If parents diagnosis are in the list of feature names,
    # check their position
    if "mo_t2d" and "fa_t2d" in feat_names:
        if feat_names[-2] != "mo_t2d" or feat_names[-1] != "fa_t2d":
            x = feat_names.index("mo_t2d")
            y = feat_names.index("fa_t2d")
            n = len(feat_names)
            raise ValueError(
                f"mo_t2d is at index {x} and fa_t2d is at index {y} from \
                    feat_names with {n} items. They must be in the \
                        penultimate and last positions, respectively."
            )


def _len_encoding(feat_names: List[str]) -> int:
    """Gets length of encoded features

    Parameters
    ----------
    feat_names : List[str]
        A list of feature names

    Returns
    -------
    int
        Length of encoded features
    """
    # If 'mo_t2d' and 'fa_t2d' header are in feature names,
    # both use 3 columns. So, +2 each one, totally 4 extra columns.
    if "mo_t2d" and "fa_t2d" in feat_names:
        return len(feat_names) + 4

    return len(feat_names)


class DiabDataset(Dataset):
    """DiabNet Dataset.

    Attributes
    ----------
    feat_names : List[str]
        Feature names.
    n_feat : int
        Number of encoded features.
    features : torch.Tensor
        Encoded features.
    labels : torch.Tensor
        Target labels.
    _raw_values : np.ndarray
        Raw feature values.
    _raw_labels : np.ndarray
        Raw target labels.
    _ages : np.ndarray
        Raw ages
    """

    def __init__(
        self,
        fn: str,
        feat_names: List[str],
        label_name: str = "T2D",
        soft_label: bool = True,
        soft_label_baseline: float = 0.0,
        soft_label_topline: float = 1.0,
        soft_label_baseline_slope: float = 0.0,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        fn : str
            Path to CSV file with data. The first line must have feature names.
        feat_names : List[str]
            A list of feature names to be used in DiabNet.
        label_name : str, optional
            Feature name (header in `fn`) that identify the target label,
            by default "T2D".
        soft_label : bool, optional
            Whether to use soft label. Negative probabilities > 0.0 and
            Positive probabilities < 1.0, by default True.
        soft_label_baseline : float, optional
            Value to negative probabilities, by default 0.0.
        soft_label_topline : float, optional
            Value to positive probabilities, by default 1.0.
        soft_label_baseline_slope : float, optional
            Decrease in negative probabilities uncertainty. Unevenness between
            20 yo and 80 yo, by default 0.0.
        device : str, optional
            Tensor device, by default "cuda".
        """
        # Check arguments
        if type(fn) not in [str]:
            raise TypeError("`fn` must be a string.")
        if not fn.endswith(".csv"):
            raise ValueError("`fn` must be a CSV-formatted file.")
        if type(label_name) not in [str]:
            raise TypeError("`label_name` must be a string.")
        if type(soft_label) not in [bool]:
            raise TypeError("`soft_label` must be a boolean.")
        if type(soft_label_baseline) not in [float]:
            raise TypeError("`soft_label_baseline` must be a floating point.")
        if type(soft_label_topline) not in [float]:
            raise TypeError("`soft_label_topline` must be a floating point.")
        if type(soft_label_baseline_slope) not in [float]:
            raise TypeError("`soft_label_baseline` must be a floating point.")
        if type(device) not in [str]:
            raise TypeError("`device` must be a string.")

        # Read data from CSV-formatted file
        df = pd.read_csv(fn)

        # Check if parents diagnosis is in the correct position
        _check_parents_diag(feat_names)

        # ! Attributes ! #
        # Feature names
        self.feat_names = feat_names
        # Number of encoded features
        self.n_feat = _len_encoding(feat_names)
        # Raw feature values
        self._raw_values = df[feat_names].values
        # Raw target labels
        self._raw_labels = df[label_name].values
        # Raw ages
        self._ages = df["AGE"].values
        # Encoded features in torch.Tensor object
        self.features = torch.tensor(
            [
                encode_features(self.feat_names, raw_value)
                for raw_value in self._raw_values
            ],
            dtype=torch.float,
        ).to(device)
        # Prepare target labels
        if soft_label:
            # Use soft labels on target labels
            labels = self._soft_label(
                baseline=soft_label_baseline,
                topline=soft_label_topline,
                baseline_slope=soft_label_baseline_slope,
            )
        else:
            # Use target labels
            labels = self._raw_labels
        # Change shape from (n, ) to (n, 1)
        self.labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), 1).to(
            device
        )

    def _soft_label(
        self, baseline: float, topline: float, baseline_slope: float
    ) -> np.ndarray:
        """Convert raw target labels to soft labels.

        Parameters
        ----------
        baseline : float
            Value to negative probabilities
        topline : float
            Value to positive probabilities
        baseline_slope : float
            Decrease in negative probabilities uncertainty. Unevenness between
            20 yo and 80 yo.

        Returns
        -------
        np.ndarray
            A numpy.ndarray with soft labels.

        Raises
        ------
        ValueError
            `soft_label_topline` must be greater than `soft_label_baseline`.
        ValueError
            `soft_label_baseline_slope` must be a negative floating point or
            zero.
        """
        # Check arguments
        if baseline >= topline:
            raise ValueError(
                "`soft_label_topline` must be greater than \
                    `soft_label_baseline`."
            )
        if baseline_slope > 0.0:
            raise ValueError(
                "`soft_label_baseline_slope` must be a negative floating point \
                    or zero."
            )

        # Correct target labels with soft labels method
        if baseline_slope == 0.0:
            soft_labels = np.maximum(self._raw_labels * topline, baseline)
        else:
            # Base adjusted considering age of the subject
            # NOTE: The younger the greater uncertainty about its negative
            #  label
            ages = np.minimum(np.maximum(self._ages, 20.0), 80.0)
            base_adjusted_by_age = np.maximum(
                baseline + baseline_slope * (ages - 20.0) / (80.0 - 20.0), 0
            )
            soft_labels = np.maximum(self._raw_labels * topline, base_adjusted_by_age)

        return soft_labels

    def __len__(self) -> int:
        """Returns the number of patients"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets encoded features and target labels"""
        return self.features[idx], self.labels[idx]

    # FIXME: Seem unused -> Remove
    # def adjust_soft_label(
    #     self, baseline: float, topline: float, baseline_slope: float
    # ) -> None:
    #     labels = self._soft_label(baseline, topline, baseline_slope)
    #     self.labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), 1).to(
    #         "cuda"
    #     )
