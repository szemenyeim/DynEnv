import numpy as np
import h5py
import best
import glob
import os
import os.path as osp
import scipy
import xlwt
import random
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)

VALS_TO_USE = 'val'

def getValues(f, robo):
    mean_pos = np.array(f['ep_pos_rewards']['mean'])
    val_pos = np.array(f['ep_pos_rewards']['values'])
    max_pos = np.array(f['ep_pos_rewards']['max'])

    mean_obs = np.array(f['ep_obs_rewards']['mean'])
    val_obs = np.array(f['ep_obs_rewards']['values'])
    max_obs = np.array(f['ep_obs_rewards']['max'])

    mean = np.array(f['ep_rewards']['mean'])
    val = np.array(f['ep_rewards']['values'])
    max = np.array(f['ep_rewards']['max'])

    mean_corr = (mean_pos if robo else mean) - mean_obs
    val_corr = (val_pos if robo else val) - val_obs
    max_corr = (max_pos if robo else max) - max_obs

    mean_corr = np.expand_dims(mean_corr,1)
    max_corr = np.expand_dims(max_corr,1)
    mean_obs = np.expand_dims(mean_obs,1)
    max_obs = np.expand_dims(max_obs,1)

    if VALS_TO_USE == 'mean':
        return mean_corr, mean_obs
    elif VALS_TO_USE == 'max':
        return max_corr, max_obs
    else:
        return val_corr, val_obs


def getAvgValues(folder):

    rews = []
    obss = []

    for file in glob.glob1(folder, "*.hdf5"):
        fname = osp.join(folder, file)
        f = h5py.File(fname, 'r')
        robo = False if "Drive" in folder else True
        rew, obs = getValues(f, robo)
        rews.append(rew)
        obss.append(obs)

    if not len(rews):
        return (np.zeros(10), np.zeros(10))

    rews = np.concatenate(rews,axis=1)
    obss = np.concatenate(obss,axis=1)

    filter = np.ones((N,1))/N

    rew_avg = scipy.signal.convolve2d(rews, filter, mode='valid')
    obs_avg = scipy.signal.convolve2d(obss, filter, mode='valid')

    rew_max = np.max(rew_avg,axis=0)
    obs_max = np.max(obs_avg,axis=0)

    return (rew_max, obs_max)


def makePlot(best_results, name, s, bins = 30):

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    #axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])

    best.plot_posterior(best_results,
                   'Difference of means',
                   ax=axes,
                   bins=bins,
                   title='Difference of means',
                   stat='mean',
                   ref_val=0,
                   label=r'$\mu_1 - \mu_2$')

    fig.tight_layout()

    fig.savefig(name + "/" + s + "_DoM.pdf")

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    best.plot_posterior(best_results,
                   'Effect size',
                   ax=axes,
                   bins=bins,
                   title='Effect size',
                   ref_val=0,
                   label=r'$(\mu_1 - \mu_2) /'
                          r' \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$')

    fig.tight_layout()

    fig.savefig(name + "/" + s + "_ES.pdf")


def getValsFromTest(best_results):

    hdi_min, hdi_max = best_results.hdi("Difference of means", 0.95)

    trace = best_results.trace['Difference of means']

    mean = trace.mean()
    median = np.median(trace)

    hist_vals, hist_bins = np.histogram(trace, bins=1000)
    pos_bins = (hist_bins >= 0)[:-1]
    ratio = np.sum(hist_vals[pos_bins]) / np.sum(hist_vals) * 100

    return hdi_min, hdi_max, ratio, mean, median


def sucStr(val):
    if val is not None:
        return "Success" if val else "Failure"
    return "Undefined"


if __name__ == '__main__':

    N = 10

    np.random.seed(42)
    random.seed(42)

    root = "Saves"

    folders = sorted(os.listdir(root))

    dict = {}

    for folder in folders:

        name = folder

        dir = osp.join(root,folder)

        if (not osp.isdir(dir)) or ("Results" in folder):
            continue

        dict[name] = getAvgValues(dir)

    comparisons = [
        ["Part","Part_Pred"],
        ["Part","Part_Pred_Rec"],
        ["Part_Pred","Part_Pred_Rec"],
        ["Full","Full_Pred"],
        ["Driving","Driving_Pred"],
    ]

    book = xlwt.Workbook(encoding="utf-8")

    styleBad = xlwt.easyxf('pattern: pattern solid, fore_colour red;')
    styleGood = xlwt.easyxf('pattern: pattern solid, fore_colour green;')
    styleUndef = xlwt.easyxf('pattern: pattern solid, fore_colour yellow;')
    style = xlwt.easyxf('font: bold 1; pattern: pattern solid, fore_colour pale_blue;')

    sheet = book.add_sheet("Basic Data")
    sheet.write(0, 0, "Metric", style)
    sheet.write(0, 1, "Rewards", style)
    sheet.write(0, 2, "Obs Rewards", style)

    for i, k in enumerate(dict.keys()):
        rew, obs = dict[k]
        sheet.write(i+1, 0, k, style)
        sheet.write(i+1, 1, rew.mean(), styleUndef)
        sheet.write(i+1, 2, obs.mean(), styleUndef)

    success = {}

    root = "Saves/Results"
    if not osp.exists(root):
        os.mkdir(root)

    for comp in comparisons:
        data2 = dict[comp[0]]
        data1 = dict[comp[1]]

        n = comp[0] +"VS" + comp[1]

        name = osp.join(root, n)
        if not osp.exists(name):
            os.mkdir(name)

        sheet = book.add_sheet(n)

        sheet.write(0, 0, "Metric", style)
        sheet.write(0, 1, "Min", style)
        sheet.write(0, 2, "Max", style)
        sheet.write(0, 3, "Mean", style)
        sheet.write(0, 4, "Median", style)
        sheet.write(0, 5, "Probability", style)

        valueStr = ["Rew", "Obs"]

        suc = []

        for i,s in enumerate(valueStr):

            if s == "Obs" and ("Full" in n or "Driv" in n):
                suc.append(None)
                continue

            print("Running BEST for", n, s)

            best_out = best.analyze_two(data1[i], data2[i])
            hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
            makePlot(best_out, name, s)

            sheet.write(i+1, 0, s, style)
            sheet.write(i+1, 1, hdi_min, styleBad if hdi_min < 0 else styleGood)
            sheet.write(i+1, 2, hdi_max, styleBad if hdi_max < 0 else styleGood)
            sheet.write(i+1, 3, mean, styleBad if mean < 0 else styleGood)
            sheet.write(i+1, 4, median, styleBad if median < 0 else styleGood)
            sheet.write(i+1, 5, ratio, styleBad if ratio < 95 else styleGood)

            suc.append(ratio > 95)

        success[n] = suc


    sheet = book.add_sheet("Summary")

    sheet.write(0, 0, "Metric", style)
    sheet.write(0, 1, "Reward", style)
    sheet.write(0, 2, "Obs Reward", style)

    line_new = '{:>25}  {:>10}  {:>10}'.format("Metric", "Reward", "Obs Reward")
    print(line_new)
    final = np.array([0,0])
    total = np.array([0,0])
    for i,suc in enumerate(success.items()):
        curr = np.array(suc[1])
        line_new = '{:>25}  {:>10}  {:>10}'.format(suc[0], sucStr(curr[0]), sucStr(curr[1]))
        print(line_new)

        sheet.write(i+1, 0, suc[0], style)
        sheet.write(i+1, 1, sucStr(curr[0]), styleGood if curr[0] else styleBad)
        sheet.write(i+1, 2, sucStr(curr[1]), styleUndef if curr[1] is None else styleGood if curr[1] else styleBad)

        total += (curr != None).astype('int64')
        curr[curr==None] = False
        final += curr.astype('int64')

    sucString = [
        str(final[0]) + "/" + str(total[0]),
        str(final[1]) + "/" + str(total[1])
    ]

    sheet.write(i+2, 0, "Success", style)
    sheet.write(i+2, 1, sucString[0], styleUndef)
    sheet.write(i+2, 2, sucString[1], styleUndef)

    line_new = '{:>25}  {:>10}  {:>10}'.format("Total", sucString[0], sucString[1])
    print(line_new)

    savePath = osp.join(root, "results.xlsx")
    book.save(savePath)
