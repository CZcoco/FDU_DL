# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    train_iters = [i for i in range(len(runner.train_loss))]
    dev_iters = [i for i in range(len(runner.dev_loss))]

    axes[0].plot(train_iters, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(dev_iters, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].legend(loc='upper right')

    train_score_iters = [i for i in range(len(runner.train_scores))]
    dev_score_iters = [i for i in range(len(runner.dev_scores))]
    axes[1].plot(train_score_iters, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(dev_score_iters, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')