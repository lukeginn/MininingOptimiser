import logging as logger
import seaborn as sns
import matplotlib.pyplot as plt


def generate_correlation_matrix(
    data, features, method="kendall", csv_path=None, plotting_path=None
):
    logger.info("Generating correlation matrix")
    data = data[features]

    correlation_matrix = create_correlation_matrix(data=data, method=method)

    if csv_path is not None:
        saving_correlation_matrix(correlation_matrix=correlation_matrix, path=csv_path)

    if plotting_path is not None:
        plot_correlation_matrix(
            correlation_matrix=correlation_matrix, path=plotting_path
        )

    return correlation_matrix


def create_correlation_matrix(data, method):
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    correlation_matrix = data.corr(method=method)
    return correlation_matrix


def saving_correlation_matrix(correlation_matrix, path):
    logger.info(f"Saving correlation matrix to {path}")
    correlation_matrix.to_csv(path)


def plot_correlation_matrix(correlation_matrix, path=None):
    logger.info(f"Plotting correlation matrix to {path}")
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 10},
    )
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
    plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
    if path:
        plt.savefig(path)
    plt.close()  # Close the plot to free up memory
