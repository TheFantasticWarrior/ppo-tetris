import matplotlib.pyplot as plt
import pandas as pd


def window_avg(data, window_width=32):
    ravgs = [sum(data[i*window_width:(i+1)*window_width]) / window_width for i in range(len(data)//window_width)]
    return ravgs
def plot_loss_data(lr,df_losses, df_data, step_column_losses, step_column_data):
    # Define the number of subplots
    num_rows = 3  # Assuming two DataFrames to plot
    num_columns = 3  # Adjust if needed

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 10))

    fig.suptitle(lr)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Plot the first DataFrame
    steps_losses = df_losses[step_column_losses]
    columns_to_plot_losses = [
        col for col in df_losses.columns if col != step_column_losses]
    for i, column in enumerate(columns_to_plot_losses):
        loss=window_avg(df_losses[column],args.nepoch*args.nminibatch)
        #loss=df_losses[column]
        axs[i].plot(loss, label=column)
        axs[i].set_title(f'{column} vs. {step_column_losses}')
        axs[i].set_xlabel(step_column_losses)
        axs[i].set_ylabel(column)
        if column == "pg_loss":
            axs[i].axhline(y=0, color='r', linestyle='-')

        axs[i].legend()

    # Plot the second DataFrame
    steps_data = df_data[step_column_data]
    columns_to_plot_data = [
        col for col in df_data.columns if col != step_column_data]
    start_index = len(columns_to_plot_losses)
    for i, column in enumerate(columns_to_plot_data, start=start_index):
        axs[i].plot(steps_data, df_data[column], label=column)
        axs[i].set_title(f'{column} vs. {step_column_data}')
        axs[i].set_xlabel(step_column_data)
        axs[i].set_ylabel(column)
        # if column == "total rewards":
        #     axs[i].axhline(y=-0.0488, color='r', linestyle='-')
        axs[i].legend()

    # Remove any unused subplots
    for i in range(len(columns_to_plot_losses) + len(columns_to_plot_data), len(axs)):
        fig.delaxes(axs[i])

    # Adjust layout to prevent overlap
    plt.tight_layout()


# Read the data from Feather files
def save_plot(lr):
    df_losses = pd.read_feather("losses.feather")
    df_data = pd.read_feather("data.feather")

# Plot the DataFrames separately within the same image
    plot_loss_data(lr,df_losses, df_data, 'step', 'iteration')
    plt.savefig(f"{lr}.png")
    plt.clf()
if __name__=="__main__":

    import args
    df_losses = pd.read_feather("losses.feather")
    df_data = pd.read_feather("data.feather")

    plot_loss_data(args.lr, df_losses, df_data, 'step', 'iteration')
    plt.show()
