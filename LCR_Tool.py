import pandas as pd
from tkinter import simpledialog
from tkinter import Scrollbar
from tkinter import Text
import locale
import subprocess
from tkinter import Tk, Toplevel, Label, Entry, Button, Text, IntVar, Scrollbar, W
import tkinter.messagebox as messagebox
import tkinter as tk
from tkinter import ttk
import tempfile
import webbrowser
from datetime import datetime
import threading
from tkinter import filedialog
import os
import re
import tkinter.font as tkfont
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import datetime64
import time
import pandas as pd
import numpy as np
import os
import time
import datetime

timestr = time.strftime("%Y%m%d-%H%M")
f = open(f"parser_error_{timestr}.txt", "a")



def get_asset_in_aggregate(df_moto):

    HLA = 0
    for ind in range(0, 4):
        HLAdat = df_moto.iloc[ind]
        try:
            HLAdat = HLAdat.astype(float)
            HLA = HLA + HLAdat
        except:
            False

    LA = 0
    for ind in range(6, 13):
        LAdat = df_moto.iloc[ind]
        try:
            LAdat = LAdat.astype(float)
            LA = LA + LAdat
        except:
            False

    IA = 0
    for ind in range(16, 19):
        IAdat = df_moto.iloc[ind]
        try:
            IAdat = IAdat.astype(float)
            IA = IA + IAdat
        except:
            False

    return HLA, LA, IA


def get_cash_inflow_aggregate(df_cashin):

    CI5 = 0
    for ind in range(0, 7):
        dat = df_cashin.iloc[ind]
        try:
            dat = dat.astype(float)
            CI5 = CI5 + dat
        except:
            False

    CI10 = 0
    for ind in range(7, 14):
        dat = df_cashin.iloc[ind]
        try:
            dat = dat.astype(float)
            CI10 = CI10 + dat
        except:
            False

    CI30 = 0
    for ind in range(14, 21):
        dat = df_cashin.iloc[ind]
        try:
            dat = dat.astype(float)
            CI30 = CI30 + dat
        except:
            False

    CI90 = 0
    for ind in range(21, 28):
        dat = df_cashin.iloc[ind]
        try:
            dat = dat.astype(float)
            CI90 = CI90 + dat
        except:
            False

    CI365 = 0
    for ind in range(28, 35):
        dat = df_cashin.iloc[ind]
        try:
            dat = dat.astype(float)
            CI365 = CI365 + dat
        except:
            False

    return CI5, CI10, CI30, CI90, CI365


def get_cash_outflow_aggregate(df_cashout):

    CO5 = 0
    for ind in range(0, 8):
        dat = df_cashout.iloc[ind]
        try:
            dat = dat.astype(float)
            CO5 = CO5 + dat
        except:
            False

    CO10 = 0
    for ind in range(8, 16):
        dat = df_cashout.iloc[ind]
        try:
            dat = dat.astype(float)
            CO10 = CO10 + dat
        except:
            False

    CO30 = 0
    for ind in range(16, 24):
        dat = df_cashout.iloc[ind]
        try:
            dat = dat.astype(float)
            CO30 = CO30 + dat
        except:
            False

    CO90 = 0
    for ind in range(24, 32):
        dat = df_cashout.iloc[ind]
        try:
            dat = dat.astype(float)
            CO90 = CO90 + dat
        except:
            False

    CO365 = 0
    for ind in range(32, 40):
        dat = df_cashout.iloc[ind]
        try:
            dat = dat.astype(float)
            CO365 = CO365 + dat
        except:
            False

    return CO5, CO10, CO30, CO90, CO365


def get_SLR_aggregate(df_moto):

    SLR5D = df_moto.iloc[23]
    SLR10D = df_moto.iloc[24]
    SLA30D = df_moto.iloc[25]
    SLA90D = df_moto.iloc[26]
    SLA365D = df_moto.iloc[27]

    return SLR5D, SLR10D, SLA30D, SLA90D, SLA365D


def cash_asset_checker(df_moto, HLA, LA, IA, df):

    time_val = df_moto.iloc[0]
    time_val = time_val.keys()
    time_val = time_val[0]

    # compare if HLA calculated match aggregate
    compare_HLA = df.iloc[16]
    if HLA == compare_HLA[0]:
        print("HLA matches")

    else:
        print(
            f"HLA doesn't match. Calculated HLA is {compare_HLA[0]} and aggregate is {HLA} on {time_val}.",
            file=f)

    #compare if LA calculated match aggregate
    compare_LA = df.iloc[22]
    
    if LA == compare_LA[0]:
        print("LA matches")

    else:
        print(f"LA doesn't match. Calculated LA is {compare_LA[0]} and aggregate is {LA} on {time_val}.", file=f)

    #compare if IA calculated match aggregate
    compare_IA = df.iloc[32]
    if IA == compare_IA[0]:
        print("IA matches")

    else:
        print(f"IA doesn't match. Calculated IA is {compare_IA[0]} and aggregate is {IA} on {time_val}", file=f)


def cash_inflow_checker(CI5, CI10, CI30, CI90, CI365, df_moto):
    time_val = df_moto.iloc[0]
    time_val = time_val.keys()
    time_val = time_val[0]
    # compare if cash inflow upto 5 days calculated match aggregate
    compare_5 = df_moto.iloc[44]
    if CI5[0] == compare_5[0]:
        print("5BD Cash Inflow matches")
    elif abs(CI5[0] - compare_5[0]) < 0.01:
        print("5BD Cash Inflow matches")
    else:
        print(f"5BD Cash Inflow doesn't match. Calculated CI5 is {CI5[0]} and aggregate is {compare_5[0]} on {time_val}.", file=f)

    # compare if cash inflow upto 10 days calculated match aggregate
    compare_10 = df_moto.iloc[45]
    if CI10[0] == compare_10[0]:
        print("10BD Cash Inflow matches")
    elif abs(CI10[0] - compare_10[0]) < 0.01:
        print("10BD Cash Inflow matches")
    else:
        print(f"10BD Cash Inflow doesn't match. Calculated CI10 is {CI10[0]} and aggregate is {compare_10[0]} on {time_val}.", file=f)

    # compare if cash inflow upto 30 days calculated match aggregate
    compare_30 = df_moto.iloc[46]
    if CI30[0] == compare_30[0]:
        print("30CD Cash Inflow matches")
    elif abs(CI30[0] - compare_30[0]) < 0.01:
        print("30CD Cash Inflow matches")
    else:
        print(f"30CD Cash Inflow doesn't match. Calculated CI30 is {CI30[0]} and aggregate is {compare_30[0]} on {time_val}.", file=f)

    # compare if cash inflow upto 90 days calculated match aggregate
    compare_90 = df_moto.iloc[47]
    if CI90[0] == compare_90[0]:
        print("90CD Cash Inflow matches")
    elif abs(CI90[0] - compare_90[0]) < 0.01:
        print("90CD Cash Inflow matches")
    else:
        print(f"90CD Cash Inflow doesn't match. Calculated CI90 is {CI90[0]} and aggregate is {compare_90[0]} on {time_val}.", file=f)

    # compare if cash inflow upto 365 days calculated match aggregate
    compare_365 = df_moto.iloc[48]
    if CI365[0] == compare_365[0]:
        print("365CD Cash Inflow matches")

    elif abs(CI365[0] - compare_365[0]) < 0.01:
        print("365CD Cash Inflow matches")

    else:
        print(f"365CD Cash Inflow doesn't match. Calculated CI365 us {CI365[0]} and aggregate is {compare_90[0]} on {time_val}.", file=f)


def cash_outflow_checker(CO5, CO10, CO30, CO90, CO365, df_moto):

    time_val = df_moto.iloc[0]
    time_val = time_val.keys()
    time_val = time_val[0]
    # threshold
    # compare if cash outflows upto 5 days calculated match aggregate
    compare_5 = df_moto.iloc[51]
    if CO5[0] == compare_5[0]:
        print("5BD Cash Outflow matches")

    elif abs(CO5[0] - compare_5[0]) < 0.01:
        print("5BD Cash Outflow matches")
    else:
        print(f"5BD Cash Outflow doesn't match. Calculated CO5 is {CO5[0]} and aggregate is {compare_5[0]} on {time_val}.", file=f)

    # compare if cash outflows upto 10 days calculated match aggregate
    compare_10 = df_moto.iloc[52]
    if CO10[0] == compare_10[0]:
        print("10BD Cash Outflow matches")
    elif abs(CO10[0] - compare_10[0]) < 0.01:
        print("10BD Cash Outflow matches")
    else:
        print(f"10BD Cash Outflow doesn't match. Calculated CO10 is {CO10[0]} and aggregate is {compare_10[0]} on {time_val}.", file=f)

    # compare if cash outflows upto 30 days calculated match aggregate
    compare_30 = df_moto.iloc[53]
    if CO30[0] == compare_30[0]:
        print("30CD Cash Outflow matches")
    elif abs(CO30[0] - compare_30[0]) < 0.01:
        print("30CD Cash Outflow matches")
    else:
        print(f"30CD Cash Outflow doesn't match. Calculated CO30 is {CO30[0]} and aggregate is {compare_30[0]} on {time_val}.", file=f)

    # compare if cash outflows upto 90 days calculated match aggregate
    compare_90 = df_moto.iloc[54]
    if CO90[0] == compare_90[0]:
        print("90CD Cash Outflow matches")
    elif abs(CO90[0] - compare_90[0]) < 0.01:
        print("90CD Cash Outflow matches")
    else:
        print(f"90CD Cash Outflow doesn't match. Calculated CO90 is {CO90[0]} and aggregate is {compare_90[0]} on {time_val}.", file=f)

    # compare if cash outflows upto 365 days calculated match aggregate
    compare_365 = df_moto.iloc[55]
    if CO365[0] == compare_365[0]:
        print("365CD Cash Outflow matches")
    elif abs(CO365[0] - compare_365[0]) < 0.01:
        print("365CD Cash Outflow matches")
    else:
        print(f"365CD Cash Outflow doesn't match. Calculated CO365 is {CO365[0]} and aggregate is {compare_90[0]} on {time_val}.", file=f)


def discounts_on_cash_assets(repo, TT1, nd_gic1, TT2, nd_gic2, passive_eq, passive_FIN, passive_FILD, ARS, active_eq,
                             active_FIN, active_FILD, active_CC, passive_cc):
    """
    calculates shock & haircuts for each cash asset classes where applied.
    """
    # LA value after shock
    LA_30_after_shock = (repo + TT1 + nd_gic1) + (passive_eq * 0.85) + (
        passive_FIN * 0.95) + (passive_FILD * 0.9) + (passive_cc *0.85)
    LA_90_after_shock = (repo + TT1 + nd_gic1 + TT2 + nd_gic2) + (
        passive_eq * 0.75) + (passive_FIN * 0.95) + (
                passive_FILD * 0.85) + (passive_cc *0.75)
    LA_365_after_shock = (repo + TT1 + nd_gic1 + TT2 + nd_gic2) + (
        passive_eq * 0.6) + (passive_FIN * 0.9) + (
                passive_FILD * 0.75) + (passive_cc*0.6)

    # IA value after shock & haircuts
    IA_365_after_shock_n_haircut = (ARS * 0.6 * 0.96) + (active_eq * 0.6 * 0.96
                                                         ) + (active_FIN * 0.9 * 0.97) + (
                active_FILD * 0.75 * 0.97) + (active_CC * 0.6 * 0.96)

    return LA_30_after_shock, LA_90_after_shock, LA_365_after_shock, IA_365_after_shock_n_haircut


def discounts_on_derivatives(currency, equity, FIN, FIND, CC):
    """
    derivative discounts calculation for SLA (shocks)
    """
    #derivative discounts by classes
    SLA_5 = (currency * 0.05) + (equity * 0.15) + (FIN * 0.05) + (FIND * 0.1) + (CC * 0.15)
    SLA_10 = (currency * 0.05) + (equity * 0.15) + (FIN * 0.05) + (FIND * 0.1) + (CC * 0.15)
    SLA_30 = (currency * 0.05) + (equity * 0.15) + (FIN * 0.05) + (FIND * 0.1) + (CC * 0.15)
    SLA_90 = (currency * 0.1) + (equity * 0.25) + (FIN * 0.05) + (FIND * 0.15) + (CC * 0.25)
    SLA_365 = (currency * 0.2) + (equity * 0.4) + (FIN * 0.1) + (FIND * 0.25) + (CC * 0.4)

    return SLA_5, SLA_10, SLA_30, SLA_90, SLA_365


def standing_liquidity_reserve_checker(df_moto, SLR5D, SLR10D, SLR30D, SLR90D, SLR365D, SLR_5,
                                       SLA_10, SLR_30, SLR_90, SLR_365):
    """
    checker function if standing liquidity reserve calculated matches the aggregated number in spreadsheet
    """

    time_val = df_moto.iloc[0]
    time_val = time_val.keys()
    time_val = time_val[0]

    if SLR5D[0] == SLR_5[0]:
      print("SLR 5 days matching")
    elif abs(SLR5D[0] - SLR_5[0]) < 0.01:
        print("SLR 5 days matching")
    else:
        print(f"SLR 5 days are not matching for {SLR5D} and {SLR_5} on {time_val}.", file=f)

    if SLR10D[0] == SLA_10[0]:
      print("SLR 10 days matching")
    elif abs(SLR10D[0] - SLA_10[0]) < 0.01:
      print("SLR 10 days matching")
    else:
        print(f"SLR 10 days are not matching for {SLR10D} and {SLA_10} on {time_val}.", file=f)

    if SLR30D[0] == SLR_30[0]:
      print("SLR 30 days matching")
    elif abs(SLR30D[0] - SLR_30[0]) < 0.01:
      print("SLR 30 days matching")
    else:
        print(f"SLR 30 days are not matching for {SLR30D} and {SLR_30} on {time_val}.", file=f)

    if SLR90D[0] == SLR_90[0]:
      print("SLR 90 days matching")
    elif abs(SLR90D[0] - SLR_90[0]) < 0.01:
      print("SLR 90 days matching")
    else:
        print(f"SLR 90 days are not matching for {SLR90D} and {SLR_90} on {time_val}.", file=f)

    if SLR365D[0] == SLR_365[0]:
      print("SLR 365 days matching")
    elif abs(SLR365D[0] - SLR_365[0]) < 0.01:
      print("SLR 365 days matching")
    else:
        print(f"SLR 365 days are not matching for {SLR365D} and {SLR_365} on {time_val}.", file=f)


def lcr_calculation(HLA, LA_30_after_shock, LA_90_after_shock, LA_365_after_shock, IA_365_after_shock_n_haircut, SLA_5,
                    SLA_10, SLA_30, SLA_90, SLA_365,
                    CI5, CI10, CI30, CI90, CI365, CO5, CO10, CO30, CO90, CO365):
    """
    LCR = (cash assets (post shock & haircuts) + cash inflows) / (standing liquidity reserve (derivative notional * shock levels) - cash outflows (in negatives))
    """

    LCR_5 = (HLA + CI5) / (SLA_5 - CO5 )
    LCR_10 = (HLA + CI5 + CI10) / (SLA_10 - (CO5 + CO10) )
    LCR_30 = (HLA + LA_30_after_shock + CI5 + CI10 + CI30) / (SLA_30 - (CO5 + CO10) -(CO30) * 1.2)
    LCR_90 = (HLA + LA_90_after_shock + CI5 + CI10 + CI30 + CI90) / (SLA_90 - (CO5 + CO10) - (CO30 + CO90) * 1.2)
    LCR_365 = (HLA + LA_365_after_shock + IA_365_after_shock_n_haircut + CI5 + CI10 + CI30 + CI90 + CI365) / (
                SLA_365 - (CO5 + CO10) -(CO30 + CO90 + CO365) * 1.2)

    return LCR_5, LCR_10, LCR_30, LCR_90, LCR_365


def lcr_checker(LCR5, LCR10, LCR30, LCR90, LCR365, LCR_5, LCR_10, LCR_30, LCR_90, LCR_365):
    """
    define LCR5-LCR365 for direct parsing of aggregates
    """
    if LCR5 == LCR_5:
        print("LCR5 matches")

    elif abs(LCR5 - LCR_5) < 0.001:
        print("LCR5 matches")
    else:
        print("LCR5 doesn't match")

    if LCR10 == LCR_10:
        print("LCR10 matches")
    elif abs(LCR10 - LCR_10) < 0.001:
        print("LCR10 matches")
    else:
        print("LCR10 doesn't match")

    if LCR30 == LCR_30:
        print("LCR30 matches")
    elif abs(LCR30 - LCR_30) < 0.001:
        print("LCR30 matches")
    else:
        print("LCR30 doesn't match")

    if LCR90 == LCR_90:
        print("LCR90 matches")
    elif abs(LCR90 - LCR_90) < 0.001:
        print("LCR90 matches")
    else:
        print("LCR90 doesn't match")

    if LCR365 == LCR_365:
        print("LCR365 matches")
    elif abs(LCR365 - LCR_365) < 0.001:
        print("LCR365 matches")
    else:
        print("LCR365 doesn't match")

#########################################################################

def main_process(file_path):

    df_temp = pd.read_excel(f'{file_path}')


    df = pd.DataFrame(df_temp)
    passive_corporate_credit = df.loc[31]
    df.drop(index=32, inplace=True)
    df.reset_index(drop=True, inplace=True)


    #CHANGE
    df.columns = df.columns.astype(str)
    #target_date = pd.to_datetime('2023-02-15').strftime('%Y-%m-%d %H:%M:%S')
    #start_column = df.columns.get_loc(target_date)
    end_column = np.where(df.dtypes == np.float64)[0][-1] + 2

    #print(start_column)
    print(end_column)
    #CHANGE


    #####
    max_col_num = len(df.columns)
    df = df.iloc[:, 2:end_column]
    #####
    #save passive IGC row & delete it.


    df = df.drop([df.index[79], df.index[87], df.index[95], df.index[103], df.index[111],
                df.index[122], df.index[131], df.index[140], df.index[149], df.index[158]])



    df.columns = [i.strftime('%Y-%m-%d') if isinstance(i, datetime.datetime) else i for i in df.columns]

    #df = df.T.drop_duplicates().T
    #df = df.iloc[:,:-1]


    #row selections for cash assets

    #index on TS - 2 = index on df
    clarified_df = df.iloc[17:73]
    cash_ins = df.iloc[75:110]
    cash_outs = df.iloc[112:157]


    #cash inflow & outflow row selection & data clean-ups from original spreadsheet
    cash_ins = cash_ins.reset_index(drop=True)
    cash_outs = cash_outs.reset_index(drop=True)
    classified = clarified_df.reset_index(drop=True)
    passive_corporate_credit = passive_corporate_credit.reset_index(drop=True)


    passive_corporate_credit = passive_corporate_credit.to_frame().T
    passive_corporate_credit.drop(passive_corporate_credit.columns[:2], axis=1, inplace=True)
    passive_corporate_credit.columns = df.columns
    print(passive_corporate_credit)

    derivative_notionals = df.loc[47:51]
    derivative_notionals = derivative_notionals.reset_index(drop=True)


    frames = []
    for elmt in df:
        df_ll = df.loc[:, [elmt]]
        df_moto = classified.loc[:, [elmt]]
        print(df_moto)
        df_cashin = cash_ins.loc[:, [elmt]]
        df_cashout = cash_outs.loc[:, [elmt]]
        passive_cc1 = passive_corporate_credit.loc[:, [elmt]]
        passive_cc2 = passive_cc1.iloc[0]
        passive_cc = float(passive_cc2.iloc[0])
        print(passive_cc)

        #cash placement
        df_notionals = derivative_notionals.loc[:, [elmt]]
        print(df_notionals)

        HLA_value = df[elmt].iloc[16]
        LA_value = df[elmt].iloc[22]
        IA_value = df[elmt].iloc[32]



        LA_30_after_shock, LA_90_after_shock, LA_365_after_shock, IA_365_after_shock_n_haircut = discounts_on_cash_assets(
            df_moto.iloc[6],
            df_moto.iloc[7], df_moto.iloc[8], df_moto.iloc[9], df_moto.iloc[10],
            df_moto.iloc[11], df_moto.iloc[12], df_moto.iloc[13], df_moto.iloc[16],
            df_moto.iloc[17], df_moto.iloc[18], df_moto.iloc[19], df_moto.iloc[20], passive_cc)

        SLR_5, SLR_10, SLR_30, SLR_90, SLR_365 = discounts_on_derivatives(df_notionals.iloc[0], df_notionals.iloc[1],
                                                                        df_notionals.iloc[2], df_notionals.iloc[3],
                                                                        df_notionals.iloc[4])

        print(df_cashout)
        # calculate cash assets
        HLA, LA, IA = get_asset_in_aggregate(df_moto)
        print("check!!!!!!!: ", LA)

        # parse SLA aggregates
        SLR5D, SLR10D, SLR30D, SLR90D, SLR365D = get_SLR_aggregate(df_moto)

        # calculated total cash inflows by timeframeB
        CI5, CI10, CI30, CI90, CI365 = get_cash_inflow_aggregate(df_cashin)

        # calculated total cash outflows by timeframe
        CO5, CO10, CO30, CO90, CO365 = get_cash_outflow_aggregate(df_cashout)


        # LCR calculation by timeframe
        LCR_5, LCR_10, LCR_30, LCR_90, LCR_365 = lcr_calculation(HLA, LA_30_after_shock, LA_90_after_shock,
                                                                LA_365_after_shock, IA_365_after_shock_n_haircut, SLR_5,
                                                                SLR_10, SLR_30, SLR_90, SLR_365,
                                                                CI5, CI10, CI30, CI90, CI365, CO5, CO10, CO30, CO90, CO365)

        cash_outflow_checker(CO5, CO10, CO30, CO90, CO365, df_moto)
        cash_inflow_checker(CI5, CI10, CI30, CI90, CI365, df_moto)
        standing_liquidity_reserve_checker(df_moto, SLR5D, SLR10D, SLR30D, SLR90D, SLR365D, SLR_5,
                                        SLR_10, SLR_30, SLR_90, SLR_365)
        cash_asset_checker(df_moto, HLA[0], LA[0], IA[0], df_ll
                        )


        # LCR numerator & denominator calculation
        lcr_5_num = (HLA + CI5)
        lcr_5_denom = (SLR_5 - CO5 * 1.2)

        lcr_10_num = (HLA + CI5 + CI10)
        lcr_10_denom = (SLR_10 - (CO5 + CO10) * 1.2)

        lcr_30_num = (HLA + LA_30_after_shock + CI5 + CI10 + CI30)
        lcr_30_denom = (SLR_30 - (CO5 + CO10 + CO30) * 1.2)

        lcr_90_num = (HLA + LA_90_after_shock + CI5 + CI10 + CI30 + CI90)
        lcr_90_denom = (SLR_90 - (CO5 + CO10 + CO30 + CO90) * 1.2)

        lcr_365_num = (HLA + LA_365_after_shock + IA_365_after_shock_n_haircut + CI5 + CI10 + CI30 + CI90 + CI365)
        lcr_365_denom = (SLR_365 - (CO5 + CO10 + CO30 + CO90 + CO365) * 1.2)

        notioxnal_total = df_notionals.iloc[0] + df_notionals.iloc[1] + df_notionals.iloc[2] + df_notionals.iloc[3] + \
                        df_notionals.iloc[4]

        # LCR capacity value calculation by timeframe
        lcr_capacity_5 = lcr_5_num - (lcr_5_denom * 1.35)
        lcr_capacity_10 = lcr_10_num - (lcr_10_denom * 1.35)
        lcr_capacity_30 = lcr_30_num - (lcr_30_denom * 1.35)
        lcr_capacity_90 = lcr_90_num - (lcr_90_denom * 1.2)


        frame = {
            f"{elmt}": [  # LCRs
                LCR_5[0], LCR_10[0], LCR_30[0], LCR_90[0], LCR_365[0],

                # LCR factors
                lcr_5_num[0], lcr_5_denom[0], lcr_10_num[0], lcr_10_denom[0], lcr_30_num[0], lcr_30_denom[0], lcr_90_num[0],
                lcr_90_denom[0],
                lcr_365_num[0], lcr_365_denom[0],

                # LCR capacity
                lcr_capacity_5[0], lcr_capacity_10[0], lcr_capacity_30[0], lcr_capacity_90[0],

                # HLA + LA + IA Before Shock & Haircuts
                (HLA[0] + LA[0] + IA[0]),

                # LA after shock
                LA_30_after_shock[0], LA_90_after_shock[0], LA_365_after_shock[0],

                # IA after shock & haircut
                IA_365_after_shock_n_haircut[0],

                # HLAs
                HLA[0], df_moto.iloc[0][0], df_moto.iloc[1][0], df_moto.iloc[2][0], df_moto.iloc[3][0],

                # LA before shock
                LA[0], df_moto.iloc[6][0], df_moto.iloc[7][0], df_moto.iloc[8][0], df_moto.iloc[9][0], df_moto.iloc[10][0],
                df_moto.iloc[11][0], df_moto.iloc[12][0], df_moto.iloc[13][0], passive_cc, 

                # IA before shock & haircut
                IA[0], df_moto.iloc[16][0], df_moto.iloc[17][0], df_moto.iloc[18][0], df_moto.iloc[19][0],
                df_moto.iloc[20][0],

                # SLA aggregates
                SLR_5[0], SLR_10[0], SLR_30[0], SLR_90[0], SLR_365[0],

                # Derivative notional before shock
                notioxnal_total[0], df_notionals.iloc[0][0], df_notionals.iloc[1][0], df_notionals.iloc[2][0],
                df_notionals.iloc[3][0], df_notionals.iloc[4][0],

                # Net Flow Before Buffer
                (CI5[0] + CO5[0]), (CI10[0] + CO10[0]), (CI30[0] + CO30[0]), (CI90[0] + CO90[0]), (CI365[0] + CO365[0]),

                # Cash Inflow Aggregated
                CI5[0], CI10[0], CI30[0], CI90[0], CI365[0],

                # Cash Outflow Aggregated
                CO5[0], CO10[0], CO30[0], CO90[0], CO365[0],

                # Cash Inflow by Category
                # 5d
                df_cashin.iloc[0][0], df_cashin.iloc[1][0], df_cashin.iloc[2][0],
                df_cashin.iloc[3][0], df_cashin.iloc[4][0], df_cashin.iloc[5][0],
                df_cashin.iloc[6][0],
                # 10d
                df_cashin.iloc[7][0], df_cashin.iloc[8][0], df_cashin.iloc[9][0],
                df_cashin.iloc[10][0], df_cashin.iloc[11][0], df_cashin.iloc[12][0],
                df_cashin.iloc[13][0],
                # 30d
                df_cashin.iloc[14][0], df_cashin.iloc[15][0], df_cashin.iloc[16][0],
                df_cashin.iloc[17][0], df_cashin.iloc[18][0], df_cashin.iloc[19][0],
                df_cashin.iloc[20][0],
                # 90d
                df_cashin.iloc[21][0], df_cashin.iloc[22][0], df_cashin.iloc[23][0],
                df_cashin.iloc[24][0], df_cashin.iloc[25][0], df_cashin.iloc[26][0],
                df_cashin.iloc[27][0],
                # 365d
                df_cashin.iloc[28][0], df_cashin.iloc[29][0], df_cashin.iloc[30][0],
                df_cashin.iloc[31][0], df_cashin.iloc[32][0], df_cashin.iloc[33][0],
                df_cashin.iloc[34][0],

                # Cash Outflow by Category
                # 5d
                df_cashout.iloc[0][0], df_cashout.iloc[1][0], df_cashout.iloc[2][0],
                df_cashout.iloc[3][0], df_cashout.iloc[4][0], df_cashout.iloc[5][0],
                df_cashout.iloc[6][0], df_cashout.iloc[7][0],
                # 10d
                df_cashout.iloc[8][0], df_cashout.iloc[9][0], df_cashout.iloc[10][0],
                df_cashout.iloc[11][0], df_cashout.iloc[12][0], df_cashout.iloc[13][0],
                df_cashout.iloc[14][0], df_cashout.iloc[15][0],
                # 30d
                df_cashout.iloc[16][0], df_cashout.iloc[17][0], df_cashout.iloc[18][0],
                df_cashout.iloc[19][0], df_cashout.iloc[20][0], df_cashout.iloc[21][0],
                df_cashout.iloc[22][0], df_cashout.iloc[23][0],
                # 90d
                df_cashout.iloc[24][0], df_cashout.iloc[25][0], df_cashout.iloc[26][0],
                df_cashout.iloc[27][0], df_cashout.iloc[28][0], df_cashout.iloc[29][0],
                df_cashout.iloc[30][0], df_cashout.iloc[31][0],
                # 365d
                df_cashout.iloc[32][0], df_cashout.iloc[33][0], df_cashout.iloc[34][0],
                df_cashout.iloc[35][0], df_cashout.iloc[36][0], df_cashout.iloc[37][0],
                df_cashout.iloc[38][0], df_cashout.iloc[39][0]

            ]

        }

        frames.append(frame)

    #back in high school, I gave haircut to one of my friends & he's been doing buzzcut until now



    index_ = [  # LCR values
        'LCR5', 'LCR10', 'LCR30', 'LCR90', 'LCR365',

        # LCR components
        'LCR5 Numerator', 'LCR5 Denominator', 'LCR10 Numerator', 'LCR10 Denominator', 'LCR30 Numerator',
        'LCR30 Denominator', 'LCR90 Numerator', 'LCR90 Denominator', 'LCR365 Numerator', 'LCR365 Denominator',

        # LCR capacities
        'LCR5 Capacity', 'LCR10 Capacity', 'LCR30 Capacity', 'LCR90 Capacity',

        # sum of cash assets before shock & haircut
        'HLA + LA + IA Before Shock and Haircut',

        # LAs after shock
        'LA30 After Shock', 'LA90 After Shock', 'LA365 After Shock',

        # IA after shock & haircut
        'IA365 After Shock and Haircut',

        # HLAs#
        'HLA', 'HLA: Cash', 'HLA: Demand Deposit', 'HLA: Redeemable GIC',
        'HLA: Other Short-Term Money Market Product <= 1D',

        # LAs#
        'LA Before Shock', 'LA Before Shock: Repo', 'LA Before Shock: Term Deposit > 1D, <= 1M',
        'LA Before Shock: Non Redeemable GIC > 1D, <=1M', 'LA Before Shock: Term Deposit > 1M, <= 3M',
        'LA Before Shock: Non Redeemable GIC > 1M, <= 3M', 'LA Before Shock: Passive Equity',
        'LA Before Shock: Passive Fixed Income Nominal', 'LA Before Shock: Passive Fixed Income Long Durational', 
        'LA Before Shock: Passive Investment Grade Credit',

        # IAs#
        'IA Before Shock and Haircut', 'IA Before Shock and Haircut: Active Absolute Return',
        'IA Before Shock and Haircut: Active Equity', 'IA: Active Fixed Income Nominal',
        'IA Before Shock and Haircut: Active Fixed Income Long Durational',
        'IA Before Shock and Haircut: Active Corporate Credit',

        # SLR aggregate
        'Standing Liquidity Reserve 5D', 'Standing Liquidity Reserve 10D', 'Standing Liquidity Reserve 30D',
        'Standing Liquidity Reserve 90D', 'Standing Liquidity Reserve 365D',

        # derivative notionals
        'Derivative Notional Before Shock', 'Derivative Notional Before Shock: Currency',
        'Derivative Notional Before Shock: Equity', 'Derivative Notional Before Shock: Fixed Income Nominal',
        'Derivative Notional Before Shock: Fixed Income Long Durational',
        'Derivative Notional Before Shock: Corporate Credit',

        # net flow before buffer
        'Net Flow After Buffer 0BD-5BD', 'Net Flow After Buffer 6BD-10BD', 'Net Flow After Buffer 11BD-30CD', 'Net Flow After Buffer 31CD-90CD', 'Net Flow After Buffer 91CD-365CD',

        # cash inflow aggregate
        'Cash Inflow 0BD-5BD', 'Cash Inflow 6BD-10BD', 'Cash Inflow 11BD-30CD', 'Cash Inflow 31CD-90CD',
        'Cash Inflow 91CD-365CD',

        # cash outflow aggregate
        'Cash Outflow After Buffer 0BD-5BD', 'Cash Outflow After Buffer 6BD-10BD', 'Cash Outflow After Buffer 11BD-30CD', 'Cash Outflow After Buffer 31CD-90CD',
        'Cash Outflow After Buffer 91CD-365CD',

        # cash inflow by category
        # 5d
        'Cash Inflow: Contributions 0BD-5BD', 'Cash Inflow: Distributions 0BD-5BD', 'Cash Inflow: Redemptions 0BD-5BD',
        'Cash Inflow: Incomes 0BD-5BD', 'Cash Inflow: Repo Cash Flow 0BD-5BD', 'Cash Inflow: FX MTM 0BD-5BD',
        'Cash Inflow: TRS MTM 0BD-5BD',
        # 10d
        'Cash Inflow: Contributions 6BD-10BD', 'Cash Inflow: Distributions 6BD-10BD', 'Cash Inflow: Redemptions 6BD-10BD',
        'Cash Inflow: Incomes 6BD-10BD', 'Cash Inflow: Repo Cash Flow 6BD-10BD', 'Cash Inflow: FX MTM 6BD-10BD',
        'Cash Inflow: TRS MTM 6BD-10BD',
        # 30d
        'Cash Inflow: Contributions 11BD-30CD', 'Cash Inflow: Distributions 11BD-30CD',
        'Cash Inflow: Redemptions 11BD-30CD', 'Cash Inflow: Incomes 11BD-30CD', 'Cash Inflow: Repo Cash Flow 11BD-30CD',
        'Cash Inflow: FX MTM 11BD-30CD', 'Cash Inflow: TRS MTM 11BD-30CD',
        # 90d
        'Cash Inflow: Contributions 31CD-90CD', 'Cash Inflow: Distributions 31CD-90CD',
        'Cash Inflow: Redemptions 31CD-90CD', 'Cash Inflow: Incomes 31CD-90CD', 'Cash Inflow: Repo Cash Flow 31CD-90CD',
        'Cash Inflow: FX MTM 31CD-90CD', 'Cash Inflow: TRS MTM 31CD-90CD',
        # 365d
        'Cash Inflow: Contributions 91CD-365CD', 'Cash Inflow: Distributions 91CD-365CD',
        'Cash Inflow: Redemptions 91CD-365CD', 'Cash Inflow: Incomes 91CD-365CD', 'Cash Inflow: Repo Cash Flow 91CD-365CD',
        'Cash Inflow: FX MTM 91CD-365CD', 'Cash Inflow: TRS MTM 91CD-365CD',

        # cash outflow by category
        # 5d
        'Cash Outflow Before Buffer: Benefit Payments 0BD-5BD', 'Cash Outflow Before Buffer: Termination Payments 0BD-5BD',
        'Cash Outflow Before Buffer: Manager Subscriptions 0BD-5BD', 'Cash Outflow Before Buffer: Capital Calls 0BD-5BD',
        'Cash Outflow Before Buffer: Expenses 0BD-5BD', 'Cash Outflow Before Buffer: Repo Cash Flow 0BD-5BD',
        'Cash Outflow Before Buffer: FX MTM 0BD-5BD', 'Cash Outflow Before Buffer: TRS MTM 0BD-5BD',
        # 10d
        'Cash Outflow Before Buffer: Benefit Payments 6BD-10BD',
        'Cash Outflow Before Buffer: Termination Payments 6BD-10BD',
        'Cash Outflow Before Buffer: Manager Subscriptions 6BD-10BD', 'Cash Outflow Before Buffer: Capital Calls 6BD-10BD',
        'Cash Outflow Before Buffer: Expenses 6BD-10BD', 'Cash Outflow Before Buffer: Repo Cash Flow 6BD-10BD',
        'Cash Outflow Before Buffer: FX MTM 6BD-10BD', 'Cash Outflow Before Buffer: TRS MTM 6BD-10BD',
        # 30d
        'Cash Outflow Before Buffer: Benefit Payments 11BD-30CD',
        'Cash Outflow Before Buffer: Termination Payments 11BD-30CD',
        'Cash Outflow Before Buffer: Manager Subscriptions 11BD-30CD',
        'Cash Outflow Before Buffer: Capital Calls 11BD-30CD', 'Cash Outflow Before Buffer: Expenses 11BD-30CD',
        'Cash Outflow Before Buffer: Repo Cash Flow 11BD-30CD', 'Cash Outflow Before Buffer: FX MTM 11BD-30CD',
        'Cash Outflow Before Buffer: TRS MTM 11BD-30CD',
        # 90d
        'Cash Outflow Before Buffer: Benefit Payments 31CD-90CD',
        'Cash Outflow Before Buffer: Termination Payments 31CD-90CD',
        'Cash Outflow Before Buffer: Manager Subscriptions 31CD-90CD',
        'Cash Outflow Before Buffer: Capital Calls 31CD-90CD', 'Cash Outflow Before Buffer: Expenses 31CD-90CD',
        'Cash Outflow Before Buffer: Repo Cash Flow 31CD-90CD', 'Cash Outflow Before Buffer: FX MTM 31CD-90CD',
        'Cash Outflow Before Buffer: TRS MTM 31CD-90CD',
        # 365d
        'Cash Outflow Before Buffer: Benefit Payments 91CD-365CD',
        'Cash Outflow Before Buffer: Termination Payments 91CD-365CD',
        'Cash Outflow Before Buffer: Manager Subscriptions 91CD-365CD',
        'Cash Outflow Before Buffer: Capital Calls 91CD-365CD', 'Cash Outflow Before Buffer: Expenses 91CD-365CD',
        'Cash Outflow Before Buffer: Repo Cash Flow 91CD-365CD', 'Cash Outflow Before Buffer: FX MTM 91CD-365CD',
        'Cash Outflow Before Buffer: TRS MTM 91CD-365CD'

    ]

    dates = []
    for col in df.columns:
        col = datetime64(col)
        dates.append(col)

    output_frame = pd.DataFrame({k:v for d in frames for k, v in d.items()})

    output_frame.index = index_
    output_frame = output_frame.set_axis(dates, axis=1)

    #output_frame = output_frame.T.drop_duplicates().T

    f.close()

    current_date = datetime.datetime.now().strftime("%Y_%m_%d")

    export_name = f"stdout_{current_date}.xlsx"
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    export_name = os.path.join(script_directory, export_name)
    

    export_name = os.path.basename(export_name)
    output_frame.to_excel(f'{export_name}')

    return [output_frame, export_name]



scenario_data = []


def apply_dark_theme():
    # Set the color scheme
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', foreground='white', background='#333333')
    style.configure('TLabel', padding=5)
    style.configure('TButton', padding=5)
    style.configure('TEntry', padding=5)
    style.configure('TFrame', background='#444444')
    style.map('TButton', background=[('active', '#666666')])



class DataHandler:
    def __init__(self):
        self.dataframe = None
        self.filename = None

    def load_dataframe(self, filename):
        self.dataframe = pd.read_excel(filename)

    def append_data(self, data):
        df = pd.DataFrame([data], index=[0])
        if self.dataframe is None:
            self.dataframe = df
        else:
            self.dataframe = pd.concat([self.dataframe, df], ignore_index=True)

    def save_dataframe(self, filename):
        if self.dataframe is not None:
            self.dataframe.to_excel(filename, index=False)
            self.filename = filename

    def export_scenario_data(self, file_name):
        df = pd.DataFrame(scenario_data)
        df.to_excel(file_name, index=False)
        print(f"Scenario data exported to {file_name}")

    #under construction 
    def export_to_excel(self, file_name, scenario_number1, scenario_number2):
        export_df = self.dataframe[["Date", "Equities %", "Fixed Income Nominal %",
                                    "Fixed Income LD %",
                                    "Investment Grade Credit %", "ARS %", "Currency %",
                                    "Equities after shock", "Fixed Income Nominal after shock",
                                    "Fixed Income LD after shock", "Investment Grade Credit after shock",
                                    "ARS after shock", "Derivative Notional Currency after shock",
                                    "Derivative Notional Fixed Income Nominal after shock",
                                    "Derivative Notional Fixed Income Long Durational after shock",
                                    "Derivative Notional Investment Grade Credit after shock",
                                    "LCR 5", "LCR 10", "LCR 30", "LCR 90", "LCR 365"]]

        export_df.columns = ["Date", "Equities %", "Fixed Income Nominal %", "Fixed Income LD %",
                             "Investment Grade Credit %", "ARS %", "Currency %",
                             "Equities after shock", "Fixed Income Nominal after shock",
                             "Fixed Income LD after shock", "Investment Grade Credit after shock",
                             "ARS after shock", "Derivative Notional Currency after shock",
                             "Derivative Notional Fixed Income Nominal after shock",
                             "Derivative Notional Fixed Income Long Durational after shock",
                             "Derivative Notional Investment Grade Credit after shock", "LCR 5", "LCR 10", "LCR 30",
                             "LCR 90", "LCR 365"]

        buffer_df = self.dataframe[["Date", "Case 1 %", "Case 2 %", "Total Cash Outflow for 5D (Case 1)",
                                    "Total Cash Outflow for 10D (Case 1)", "Total Cash Outflow for 30D (Case 1)",
                                    "Total Cash Outflow for 90D (Case 1)", "Total Cash Outflow for 365D (Case 1)",
                                    "Total Cash Outflow for 5D (Case 2)",
                                    "Total Cash Outflow for 10D (Case 2)", "Total Cash Outflow for 30D (Case 2)",
                                    "Total Cash Outflow for 90D (Case 2)", "Total Cash Outflow for 365D (Case 2)",

                                    "LCR 5 (Case 1)", "LCR 10 (Case 1)", "LCR 30 (Case 1)", "LCR 90 (Case 1)",
                                    "LCR 365 (Case 1)",
                                    "LCR 5 (Case 2)", "LCR 10 (Case 2)", "LCR 30 (Case 2)", "LCR 90 (Case 2)",
                                    "LCR 365 (Case 2)"
                                    ]]

        buffer_df.columns = ["Date", "Case 1 %", "Case 2 %", "Total Cash Outflow for 5D (Case 1)",
                             "Total Cash Outflow for 10D (Case 1)", "Total Cash Outflow for 30D (Case 1)",
                             "Total Cash Outflow for 90D (Case 1)", "Total Cash Outflow for 365D (Case 1)",
                             "Total Cash Outflow for 5D (Case 2)",
                             "Total Cash Outflow for 10D (Case 2)", "Total Cash Outflow for 30D (Case 2)",
                             "Total Cash Outflow for 90D (Case 2)", "Total Cash Outflow for 365D (Case 2)",

                             "LCR 5 (Case 1)", "LCR 10 (Case 1)", "LCR 30 (Case 1)", "LCR 90 (Case 1)",
                             "LCR 365 (Case 1)",
                             "LCR 5 (Case 2)", "LCR 10 (Case 2)", "LCR 30 (Case 2)", "LCR 90 (Case 2)",
                             "LCR 365 (Case 2)"
                             ]

        export_df.insert(0, "Scenario", "Scenario " + str(scenario_number1))
        export_df["Date"] = pd.to_datetime(export_df["Date"]).dt.strftime("%Y-%m-%d %H:%M")

        buffer_df.insert(0, "Scenario", "Scenario " + str(scenario_number2))
        buffer_df["Date"] = pd.to_datetime(buffer_df["Date"]).dt.strftime("%Y-%m-%d %H:%M")

        export_df["Equities after shock"] = export_df["Equities after shock"].apply(lambda x: "{:,.0f}".format(x))
        export_df["Fixed Income Nominal after shock"] = export_df["Fixed Income Nominal after shock"].apply(
            lambda x: "{:,.0f}".format(x))
        export_df["Fixed Income LD after shock"] = export_df["Fixed Income LD after shock"].apply(
            lambda x: "{:,.0f}".format(x))
        export_df["Investment Grade Credit after shock"] = export_df["Investment Grade Credit after shock"].apply(
            lambda x: "{:,.0f}".format(x))
        export_df["ARS after shock"] = export_df["ARS after shock"].apply(lambda x: "{:,.0f}".format(x))
        export_df["Derivative Notional Currency after shock"] = export_df[
            "Derivative Notional Currency after shock"].apply(lambda x: "{:,.0f}".format(x))
        export_df["Derivative Notional Fixed Income Nominal after shock"] = export_df[
            "Derivative Notional Fixed Income Nominal after shock"].apply(lambda x: "{:,.0f}".format(x))
        export_df["LCR 5"] = export_df["LCR 5"].apply(lambda x: "{:,.2f}".format(x))
        export_df["LCR 10"] = export_df["LCR 10"].apply(lambda x: "{:,.2f}".format(x))
        export_df["LCR 30"] = export_df["LCR 30"].apply(lambda x: "{:,.2f}".format(x))
        export_df["LCR 90"] = export_df["LCR 90"].apply(lambda x: "{:,.2f}".format(x))
        export_df["LCR 365"] = export_df["LCR 365"].apply(lambda x: "{:,.2f}".format(x))

        buffer_df["Case 1 %"] = buffer_df["Case 1 %"].apply(lambda x: "{:,.0f}".format(x))
        buffer_df["Case 2 %"] = buffer_df["Case 2 %"].apply(lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 5D (Case 1)"] = buffer_df["Total Cash Outflow for 5D (Case 1)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 10D (Case 1)"] = buffer_df["Total Cash Outflow for 10D (Case 1)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 30D (Case 1)"] = buffer_df["Total Cash Outflow for 30D (Case 1)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 90D (Case 1)"] = buffer_df["Total Cash Outflow for 90D (Case 1)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 365D (Case 1)"] = buffer_df["Total Cash Outflow for 365D (Case 1)"].apply(
            lambda x: "{:,.0f}".format(x))

        buffer_df["Total Cash Outflow for 5D (Case 2)"] = buffer_df["Total Cash Outflow for 5D (Case 2)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 10D (Case 2)"] = buffer_df["Total Cash Outflow for 10D (Case 2)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 30D (Case 2)"] = buffer_df["Total Cash Outflow for 30D (Case 2)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 90D (Case 2)"] = buffer_df["Total Cash Outflow for 90D (Case 2)"].apply(
            lambda x: "{:,.0f}".format(x))
        buffer_df["Total Cash Outflow for 365D (Case 2)"] = buffer_df["Total Cash Outflow for 365D (Case 2)"].apply(
            lambda x: "{:,.0f}".format(x))

        buffer_df["LCR 5 (Case 1)"] = buffer_df["LCR 5 (Case 1)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 10 (Case 1)"] = buffer_df["LCR 10 (Case 1)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 30 (Case 1)"] = buffer_df["LCR 30 (Case 1)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 90 (Case 1)"] = buffer_df["LCR 90 (Case 1)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 365 (Case 1)"] = buffer_df["LCR 365 (Case 1)"].apply(lambda x: "{:,.2f}".format(x))

        buffer_df["LCR 5 (Case 2)"] = buffer_df["LCR 5 (Case 2)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 10 (Case 2)"] = buffer_df["LCR 10 (Case 2)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 30 (Case 2)"] = buffer_df["LCR 30 (Case 2)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 90 (Case 2)"] = buffer_df["LCR 90 (Case 2)"].apply(lambda x: "{:,.2f}".format(x))
        buffer_df["LCR 365 (Case 2)"] = buffer_df["LCR 365 (Case 2)"].apply(lambda x: "{:,.2f}".format(x))

        file_name = f"Scenario_{scenario_number1}_{file_name}"
        export_df = export_df.set_index("Date")
        buffer_df = buffer_df.set_index("Date")

        with pd.ExcelWriter(file_name) as writer:
            export_df.T.to_excel(writer, sheet_name="Market Shock Transposed")
            buffer_df.T.to_excel(writer, sheet_name="Cash Outflow Buffer Transposed")

        print(f"Data exported to {file_name}")


class MainMenu(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("LCR Scenarios")
        self.geometry("500x500")

        self.year = tk.StringVar()
        self.month = tk.StringVar()
        self.day = tk.StringVar()
        self.data_handler = DataHandler()

        # label = tk.Label(self, text="LCR Scenario Tool", font=("Arial", 16))
        # label.grid(row=0, column=0, columnspan=2, pady=10)

        # Year Entry
        year_label = tk.Label(self, text="Year (yyyy):")
        year_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
        year_entry = tk.Entry(self, textvariable=self.year)
        year_entry.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        # Month Entry
        month_label = tk.Label(self, text="Month (mm):")
        month_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)
        month_entry = tk.Entry(self, textvariable=self.month)
        month_entry.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

        # Day Entry
        day_label = tk.Label(self, text="Day (dd):")
        day_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)
        day_entry = tk.Entry(self, textvariable=self.day)
        day_entry.grid(row=3, column=1, padx=10, pady=5, sticky='ew')

        # Error Label
        self.error_label = tk.Label(self, text="", fg="red")
        self.error_label.grid(row=4, column=0, columnspan=2, pady=5)

        # Enter Button
        enter_button = tk.Button(self, text="Enter", command=self.submit_date, width=10)
        enter_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.bind("<Return>", lambda event: enter_button.invoke())

        # Quit Button
        quit_button = tk.Button(self, text="Quit", command=self.quit, width=10)
        quit_button.grid(row=6, column=0, columnspan=2, pady=10)
        self.bind("<Escape>", lambda event: quit_button.invoke())

        load_file_button = tk.Button(self, text="Select File", command=self.load_file, width=10)
        load_file_button.grid(row=1, column=0, columnspan=2, pady=10, padx=20, sticky='w')

        self.selected_file_label = tk.Label(self, text="", font=("Arial", 10))
        self.selected_file_label.grid(row=4, column=0, columnspan=2, pady=5, padx=20, sticky='w')

        run_parser_button = tk.Button(self, text="Process Data", command=self.run_parser, width=10)
        run_parser_button.grid(row=2, column=0, columnspan=2, pady=10, padx=20, sticky='w')

        self.status_label = tk.Label(self, text="", fg="black")
        self.status_label.grid(row=7, column=0, pady=5)

        # Move other elements to the right side
        # label.grid(row=0, column=0, pady=10)
        year_label.grid(row=1, column=2, padx=10, pady=5, sticky=tk.E)
        year_entry.grid(row=1, column=3, padx=10, pady=5, sticky='ew')
        month_label.grid(row=2, column=2, padx=10, pady=5, sticky=tk.E)
        month_entry.grid(row=2, column=3, padx=10, pady=5, sticky='ew')
        day_label.grid(row=3, column=2, padx=10, pady=5, sticky=tk.E)
        day_entry.grid(row=3, column=3, padx=10, pady=5, sticky='ew')
        enter_button.grid(row=4, column=2, columnspan=2, pady=10)
        quit_button.grid(row=5, column=2, columnspan=2, pady=10)

        comment_label = tk.Label(self, text="*Please run parser once per file*", fg="red")
        comment_label.grid(row=9, column=0, columnspan=2, pady=10,
                           sticky="se")  # Sticky "se" aligns label to the bottom-right corner

    def load_file(self):
        file_path = filedialog.askopenfilename(title="Select an .xlsx file", filetypes=[("Excel files", "*.xlsm")])
        print(file_path)
        if file_path:
            self.data_handler.load_dataframe(file_path)
            print("File loaded:", file_path)
            self.selected_file_label.config(text="Selected File: " + os.path.basename(file_path))
            self.file_path = os.path.basename(file_path)
            self.error_label.config(text="")  # Clear any error messages
            print(file_path)

    def run_parser(self):
        # Implement the functionality to run the parser here

        timestr = time.strftime("%Y%m%d-%H%M")
        f = open(f"parser_error_{timestr}.txt", "a")

        result = main_process(self.file_path)

        self.file_to_use = result[0]
        self.export_name = result[1]

        self.full_export_path = os.path.abspath(self.export_name)
        self.full_export_path = self.full_export_path.replace('\\\\', '\\')
        print(self.full_export_path)

        time.sleep(1)
        self.status_label.config(text="Parsing Complete", fg="green")

    def run_parser_with_progress(self):
        pass
        # ... (existing run_parser_with_progress code)

        # # Yield parsing status along with progress updates
        # parsing_status = "Parser in progress..." if self.file_to_use and self.export_name else "Parser not started"
        # for current_step in range(total_steps):
        #     progress_percentage = (current_step / total_steps) * 100
        #     yield progress_percentage, parsing_status
        #
        # # Yield 100% progress and completion status
        # yield 100, "Parser completed" if self.file_to_use and self.export_name else "Parser not started"
        #

    def submit_date(self):
        year = self.year.get()
        month = self.month.get()
        day = self.day.get()

        try:
            datetime_obj = datetime.datetime(int(year), int(month), int(day), 0, 0)
            expected_date_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

            #  # Get the list of .xlsx files in the current directory
            # files = [f for f in os.listdir() if f.endswith('.xlsx')]
            # #
            #  # Sort files based on the datetime in the file name
            # sorted_files = sorted(files, key=lambda x: re.search(r'std_out_(\d{4}_\d{2}_\d{2})', x).group(1), reverse=True)

            if self.export_name:
                # latest_file = sorted_files[0]
                self.data_handler.load_dataframe(self.full_export_path)

                # Compare expected_date_str with th e header columns
                headers = [str(header) for header in self.data_handler.dataframe.columns]
                if expected_date_str in headers:
                    print("Date Found in Header. Proceeding to Next Page...")
                    self.withdraw()  # Hide the MainMenu window
                    self.create_start_page(datetime_obj)  # Call the method to create StartPage
                else:
                    error_message = "Error: Date Not Found in Header. Please select a different date."
                    self.error_label.config(text=error_message)
            else:
                error_message = "Error: No .xlsx files found in the directory."
                self.error_label.config(text=error_message)
        except ValueError:
            error_message = "Error: Invalid Date Format. Please enter a valid date."
            self.error_label.config(text=error_message)

    def create_start_page(self, date_time_obj):
        start_page = StartPage(self, date_time_obj, self.data_handler)
        start_page.mainloop()


class StartPage(tk.Toplevel):
    def __init__(self, parent, date_time_obj, data_handler):
        tk.Toplevel.__init__(self, parent)
        self.title("Scenarios")
        self.date_time_obj = date_time_obj
        self.data_handler = data_handler
        self.expected_date_str = date_time_obj
        self.geometry("1650x500")  # Adjust the size as needed

        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        label = tk.Label(button_frame, text="Scenarios", font=("Arial", 16))
        label.pack(pady=10, padx=10)

        button1 = tk.Button(button_frame, text="Reallocation", width=20, height=1, command=self.open_reallocation)
        button3 = tk.Button(button_frame, text="Cash Outflow Buffer", width=20, height=1,
                            command=self.open_cash_outflow_buffer)
        button4 = tk.Button(button_frame, text="Market Shock", width=20, height=1, command=self.open_market_shock)
        button5 = tk.Button(button_frame, text="Shock/Haircut Change", width=20, height=1,
                            command=self.open_shock_haircut)
        button6 = tk.Button(button_frame, text="Connect DOMO", width=20, height=1, command=self.connect_to_domo)
        button7 = tk.Button(button_frame, text="Select Other Date", width=20, height=1, command=self.back_to_main_menu)

        # button1 = tk.Button(self, text="Reallocation", width=20, height=1, command=self.open_reallocation)
        # button3 = tk.Button(self, text="Cash Outflow Buffer", width=20, height=1, command=self.open_cash_outflow_buffer)
        # button4 = tk.Button(self, text="Market Shock", width=20, height=1, command=self.open_market_shock)
        # button5 = tk.Button(self, text="Shock/Haircut Change", width=20, height=1, command=self.open_shock_haircut)
        # #button6 = tk.Button(self, text="New Organization", width=20, height=1, command=self.open_new_organization)
        # button7 = tk.Button(self, text="Back to Main Menu", command=self.back_to_main_menu, width=20, height=1)

        button1.pack()
        button3.pack()
        button4.pack()
        button5.pack()
        button6.pack()
        button7.pack()

        treeview_frame = tk.Frame(self)
        treeview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.date_time_obj]

        locale.setlocale(locale.LC_ALL, '')
        # Create the dataframe
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market', '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit (1D-1M)', 'Non-Redeemable GICs (1D-1M)',
                      'Term Deposits (1M-3M)',
                      "Non-Redeemable GICs (1M-3M)", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6), round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal','FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" if isinstance(x, int) or isinstance(x,
                                                                                                                float) else x)

        # Create Treeview
        self.treeview = ttk.Treeview(treeview_frame, columns=list(df.columns), show='headings')
        for col in df.columns:
            self.treeview.heading(col, text=col, anchor='w')
            self.treeview.column(col, width=120, anchor='w')

        for _, row in df.iterrows():
            self.treeview.insert('', 'end', values=list(row))

        seperator = ['-----------------------', '-----------------------', 
                     '-----------------------',
                     '-----------------------', '-----------------------', 
                     '-----------------------',
                     '-----------------------', '-----------------------', 
                     '-----------------------',
                     '-----------------------', '-----------------------',
                     '-----------------------', '-----------------------', 
                     '-----------------------']
        self.treeview.insert('', 'end', values=seperator)

        slr_lcr_row = ['<SLRs>', '<SLR Values>', '<Cash Inflows>', '<CI Values>', 
                       '<Cash Outflows>', '<CO Values>',
                       '<LCRs>', '<LCR Values>', '', '', '', '', '', '', '']
        self.treeview.insert('', 'end', values=slr_lcr_row)

        # Insert a new row for LCRs
        row_5d = ['SLR 0-5D', '$' + str(round(o[45] / 1e6)) + 'm', 'CI 0-5D', '$' + str(
            round(o[61] / 1e6)) + 'm',
                  'CO 0-5D', ('-' if round(o[66] / 1e6) < 0 else '') + '$' + str(
            abs(round(o[66] / 1e6))) + 'm',
                  'LCR 5D', round(o[0], 2), '', '', '', '', '', '']
        self.treeview.insert('', 'end', values=row_5d)

        row_10d = ['SLR 6-10D', '$' + str(round(o[46] / 1e6)) + 'm', 'CI 6-10D', '$' + str(
            round(o[62] / 1e6)) + 'm',
                   'CO 6-10D', ('-' if round(o[67] / 1e6) < 0 else '') + '$' + str(
            abs(round(o[67] / 1e6))) + 'm',
                   'LCR 10D', round(o[1], 2), '', '', '', '', '', '']
        self.treeview.insert('', 'end', values=row_10d)

        row_30d = ['SLR 11-30D', '$' + str(round(o[47] / 1e6)) + 'm', 'CI 11-30D', '$' + str(
            round(o[63] / 1e6)) + 'm',
                   'CO 11-30D', ('-' if round(o[68] / 1e6) < 0 else '') + '$' + str(
            abs(round(o[68] / 1e6))) + 'm',
                   'LCR 30D', round(o[2], 2), '', '', '', '', '', '']
        self.treeview.insert('', 'end', values=row_30d)

        row_90d = ['SLR 31-90D', '$' + str(round(o[48] / 1e6)) + 'm', 'CI 31-90D', '$' + str(
            round(o[64] / 1e6)) + 'm',
                   'CO 31-90D', ('-' if round(o[69] / 1e6) < 0 else '') + '$' + str(
            abs(round(o[69] / 1e6))) + 'm',
                   'LCR 90D', round(o[3], 2), '', '', '', '', '', '']
        self.treeview.insert('', 'end', values=row_90d)

        row_365d = ['SLR 91-365D', '$' + str(round(o[49] / 1e6)) + 'm', 'CI 91-365D',
                    '$' + str(round(o[65] / 1e6)) + 'm',
                    'CO 91-365D', ('-' if round(o[70] / 1e6) < 0 else '') + '$' + str(
            abs(round(o[70] / 1e6))) + 'm',
                    'LCR 365D', round(o[4], 2), '', '', '', '', '',
                    '']
        self.treeview.insert('', 'end', values=row_365d)

        self.treeview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # cash ops (date in the timeframe)
    # comments in input field (market shock -> currencies = derivatives; ); comment on distinguishing shocks (incremental vs not)
    # screen for portfolio statistics

    def open_cash_outflow_buffer(self):
        cash_outflow_buffer_page = CashOutflowBuffer(self, self.date_time_obj, 
                                                     self.data_handler)
        cash_outflow_buffer_page.mainloop()

    def open_reallocation(self):
        reallocation_page = Reallocation(self, self.date_time_obj, self.data_handler)
        reallocation_page.mainloop()

    def open_shock_haircut(self):
        shock_haircut_page = Shock_Haircut_Change(self, self.date_time_obj, self.data_handler)
        shock_haircut_page.mainloop()

    # def open_new_organization(self):
    #     new_organization_page = New_Organization(self, self.date_time_obj, self.data_handler)
    #     new_organization_page.mainloop()

    def open_market_shock(self):
        market_shock_page = MarketShock(self, self.date_time_obj, self.data_handler)
        market_shock_page.mainloop()

    def connect_to_domo(self):
        domo_url = "https://universitypensionplan-ca.domo.com/page/402924416"
        webbrowser.open_new_tab(domo_url)

    def connect_to_bloomberg(self):
        possible_paths = [
            r"C:\Program Files\Bloomberg\Terminal\Bloomberg.exe",  # Example path
            r"C:\Program Files (x86)\Bloomberg\Terminal\Bloomberg.exe"  
            # Example path for 32-bit system
            # Add more paths as needed
        ]

        found_executable = False

        for path in possible_paths:
            if os.path.exists(path):
                found_executable = True
                try:
                    subprocess.Popen(path)
                except:
                    messagebox.showerror("Error", 
                                         "An error occurred while trying to open Bloomberg.")
                break

        if not found_executable:
            messagebox.showerror("Error", "Bloomberg executable not found.")

    def back_to_main_menu(self):
        save_data = messagebox.askyesno("Save Data", "Do you want to save the data?")

        if save_data:
            current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
            file_name = f"data_{current_datetime}.xlsx"
            self.data_handler.export_scenario_data(file_name)

        self.destroy()  # Close the StartPage window
        self.master.deiconify()


class Reallocation(tk.Toplevel):
    def __init__(self, parent, expected_date_str, data_handler):
        tk.Toplevel.__init__(self, parent)
        self.title("Reallocations")
        self.expected_date_str = expected_date_str
        self.data_handler = data_handler
        self.geometry("1750x700")
        # label = tk.Label(self, text="Reallocations", font=("Arial", 16))
        # label.grid(row=0, column=0, pady=10, padx=10, columnspan=12)

        # HLAs
        self.cash = tk.IntVar()
        self.demandDeposit = tk.IntVar()
        self.redeemableGIC = tk.IntVar()
        self.short_term_money_market = tk.IntVar()

        # LAs
        self.repo = tk.IntVar()
        self.termDeposit_1d_1m = tk.IntVar()
        self.non_redeemableGIC_1d_1m = tk.IntVar()
        self.termDeposit_1m_3m = tk.IntVar()
        self.non_redeemableGIC_1m_3m = tk.IntVar()
        self.passiveEQ = tk.IntVar()
        self.passiveFIN = tk.IntVar()
        self.passiveFILD = tk.IntVar()
        self.passiveIGC = tk.IntVar()

        # IAs
        self.activeARS = tk.IntVar()
        self.activeEQ = tk.IntVar()
        self.activeFIN = tk.IntVar()
        self.activeFILD = tk.IntVar()
        self.corporateCredit = tk.IntVar()

        # privates
        self.activePrivateEQ = tk.IntVar()
        self.activeREITs = tk.IntVar()
        self.infra = tk.IntVar()
        self.privateDebt = tk.IntVar()

        # derivative notionals
        self.DN_currencies = tk.IntVar()
        self.DN_equities = tk.IntVar()
        self.DN_fin = tk.IntVar()
        self.DN_fild = tk.IntVar()
        self.DN_igc = tk.IntVar()

        # Label for adjustment instruction
        adjustment_label = tk.Label(self, text="Highly Liquid Assets", font=("Arial", 12, "bold"))
        adjustment_label.grid(row=1, column=0, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adjustment_label = tk.Label(self, text="Liquid Assets", font=("Arial", 12, "bold"))
        adjustment_label.grid(row=1, column=2, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adjustment_label = tk.Label(self, text="Illiquid Assets", font=("Arial", 12, "bold"))
        adjustment_label.grid(row=1, column=4, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adjustment_label = tk.Label(self, text="Private Assets", font=("Arial", 12, "bold"))
        adjustment_label.grid(row=1, column=6, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adjustment_label = tk.Label(self, text="Derivative Notionals", font=("Arial", 12, "bold"))
        adjustment_label.grid(row=1, column=8, pady=5, padx=10, sticky=tk.W, columnspan=4)

        # HLAs
        input_label1 = tk.Label(self, text="Cash")
        input_label1.grid(row=2, column=0, pady=5, padx=10, sticky=tk.W)
        input_entry1 = tk.Entry(self, textvariable=self.cash)
        input_entry1.grid(row=2, column=1, pady=5, padx=10, sticky=tk.W)

        input_label2 = tk.Label(self, text="Demand Deposit")
        input_label2.grid(row=3, column=0, pady=5, padx=10, sticky=tk.W)
        input_entry2 = tk.Entry(self, textvariable=self.demandDeposit)
        input_entry2.grid(row=3, column=1, pady=5, padx=10, sticky=tk.W)

        input_label3 = tk.Label(self, text="Redeemable GICs")
        input_label3.grid(row=4, column=0, pady=5, padx=10, sticky=tk.W)
        input_entry3 = tk.Entry(self, textvariable=self.redeemableGIC)
        input_entry3.grid(row=4, column=1, pady=5, padx=10, sticky=tk.W)

        input_label4 = tk.Label(self, text="Short-Term Money Market Products")
        input_label4.grid(row=5, column=0, pady=5, padx=10, sticky=tk.W)
        input_entry4 = tk.Entry(self, textvariable=self.short_term_money_market)
        input_entry4.grid(row=5, column=1, pady=5, padx=10, sticky=tk.W)



        # LAs
        input_label5 = tk.Label(self, text="Repos")
        input_label5.grid(row=2, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry5 = tk.Entry(self, textvariable=self.repo)
        input_entry5.grid(row=2, column=3, pady=5, padx=10, sticky=tk.W)

        input_labela = tk.Label(self, text="Term Deposit >1D <=1M")
        input_labela.grid(row=3, column=2, pady=5, padx=10, sticky=tk.W)
        input_entrya = tk.Entry(self, textvariable=self.termDeposit_1d_1m)
        input_entrya.grid(row=3, column=3, pady=5, padx=10, sticky=tk.W)

        input_label6 = tk.Label(self, text="Non-Redeemable GICs >1D <=1M")
        input_label6.grid(row=4, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry6 = tk.Entry(self, textvariable=self.non_redeemableGIC_1d_1m)
        input_entry6.grid(row=4, column=3, pady=5, padx=10, sticky=tk.W)

        input_labelb = tk.Label(self, text="Term Deposits >1M <=3M")
        input_labelb.grid(row=5, column=2, pady=5, padx=10, sticky=tk.W)
        input_entryb = tk.Entry(self, textvariable=self.termDeposit_1m_3m)
        input_entryb.grid(row=5, column=3, pady=5, padx=10, sticky=tk.W)

        input_labelc = tk.Label(self, text="Non-Redeemable GICs >1M <=3M")
        input_labelc.grid(row=6, column=2, pady=5, padx=10, sticky=tk.W)
        input_entryc = tk.Entry(self, textvariable=self.non_redeemableGIC_1m_3m)
        input_entryc.grid(row=6, column=3, pady=5, padx=10, sticky=tk.W)

        input_label7 = tk.Label(self, text="Passive Equities")
        input_label7.grid(row=7, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry7 = tk.Entry(self, textvariable=self.passiveEQ)
        input_entry7.grid(row=7, column=3, pady=5, padx=10, sticky=tk.W)

        input_label8 = tk.Label(self, text="Passive Fixed Income Nominal")
        input_label8.grid(row=8, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry8 = tk.Entry(self, textvariable=self.passiveFIN)
        input_entry8.grid(row=8, column=3, pady=5, padx=10, sticky=tk.W)

        input_label9 = tk.Label(self, text="Passive Fixed Income Long Durational")
        input_label9.grid(row=9, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry9 = tk.Entry(self, textvariable=self.passiveFILD)
        input_entry9.grid(row=9, column=3, pady=5, padx=10, sticky=tk.W)

        input_label9_1 = tk.Label(self, text="Passive Corporate Credit")
        input_label9_1.grid(row=10, column=2, pady=5, padx=10, sticky=tk.W)
        input_entry9_1 = tk.Entry(self, textvariable=self.passiveIGC)
        input_entry9_1.grid(row=10, column=3, pady=5, padx=10, sticky=tk.W)

        # IAs
        input_label10 = tk.Label(self, text="Active Absolute Return Strategies")
        input_label10.grid(row=2, column=4, pady=5, padx=10, sticky=tk.W)
        input_entry10 = tk.Entry(self, textvariable=self.activeARS)
        input_entry10.grid(row=2, column=5, pady=5, padx=10, sticky=tk.W)

        input_label11 = tk.Label(self, text="Active Equities")
        input_label11.grid(row=3, column=4, pady=5, padx=10, sticky=tk.W)
        input_entry11 = tk.Entry(self, textvariable=self.activeEQ)
        input_entry11.grid(row=3, column=5, pady=5, padx=10, sticky=tk.W)

        input_label12 = tk.Label(self, text="Active Fixed Income Nominals")
        input_label12.grid(row=4, column=4, pady=5, padx=10, sticky=tk.W)
        input_entry12 = tk.Entry(self, textvariable=self.activeFIN)
        input_entry12.grid(row=4, column=5, pady=5, padx=10, sticky=tk.W)

        input_label13 = tk.Label(self, text="Active Fixed Income Long Durationals")
        input_label13.grid(row=5, column=4, pady=5, padx=10, sticky=tk.W)
        input_entry13 = tk.Entry(self, textvariable=self.activeFILD)
        input_entry13.grid(row=5, column=5, pady=5, padx=10, sticky=tk.W)

        input_label14 = tk.Label(self, text="Active Corporate Credits")
        input_label14.grid(row=6, column=4, pady=5, padx=10, sticky=tk.W)
        input_entry14 = tk.Entry(self, textvariable=self.corporateCredit)
        input_entry14.grid(row=6, column=5, pady=5, padx=10, sticky=tk.W)

        # privates
        input_label15 = tk.Label(self, text="Active Private Equities")
        input_label15.grid(row=2, column=6, pady=5, padx=10, sticky=tk.W)
        input_entry15 = tk.Entry(self, textvariable=self.activePrivateEQ)
        input_entry15.grid(row=2, column=7, pady=5, padx=10, sticky=tk.W)

        input_label16 = tk.Label(self, text="Active Real Estates")
        input_label16.grid(row=3, column=6, pady=5, padx=10, sticky=tk.W)
        input_entry16 = tk.Entry(self, textvariable=self.activeREITs)
        input_entry16.grid(row=3, column=7, pady=5, padx=10, sticky=tk.W)

        input_label17 = tk.Label(self, text="Infrastructures")
        input_label17.grid(row=4, column=6, pady=5, padx=10, sticky=tk.W)
        input_entry17 = tk.Entry(self, textvariable=self.infra)
        input_entry17.grid(row=4, column=7, pady=5, padx=10, sticky=tk.W)

        input_label18 = tk.Label(self, text="Private Debts")
        input_label18.grid(row=5, column=6, pady=5, padx=10, sticky=tk.W)
        input_entry18 = tk.Entry(self, textvariable=self.privateDebt)
        input_entry18.grid(row=5, column=7, pady=5, padx=10, sticky=tk.W)

        # derivative notionals
        input_label19 = tk.Label(self, text="Currencies")
        input_label19.grid(row=2, column=8, pady=5, padx=10, sticky=tk.W)
        input_entry19 = tk.Entry(self, textvariable=self.DN_currencies)
        input_entry19.grid(row=2, column=9, pady=5, padx=10, sticky=tk.W)

        input_label20 = tk.Label(self, text="Equities")
        input_label20.grid(row=3, column=8, pady=5, padx=10, sticky=tk.W)
        input_entry20 = tk.Entry(self, textvariable=self.DN_equities)
        input_entry20.grid(row=3, column=9, pady=5, padx=10, sticky=tk.W)

        input_label21 = tk.Label(self, text="Fixed Income Nominal")
        input_label21.grid(row=4, column=8, pady=5, padx=10, sticky=tk.W)
        input_entry21 = tk.Entry(self, textvariable=self.DN_fin)
        input_entry21.grid(row=4, column=9, pady=5, padx=10, sticky=tk.W)

        input_label22 = tk.Label(self, text="Fixed Income Long Durational")
        input_label22.grid(row=5, column=8, pady=5, padx=10, sticky=tk.W)
        input_entry22 = tk.Entry(self, textvariable=self.DN_fild)
        input_entry22.grid(row=5, column=9, pady=5, padx=10, sticky=tk.W)

        input_label23 = tk.Label(self, text="Investment Grade Credit")
        input_label23.grid(row=6, column=8, pady=5, padx=10, sticky=tk.W)
        input_entry23 = tk.Entry(self, textvariable=self.DN_igc)
        input_entry23.grid(row=6, column=9, pady=5, padx=10, sticky=tk.W)

        # additional formating
        submit_button = tk.Button(self, text="Enter", command=self.submit_inputs, width=10)
        submit_button.grid(row=11, column=0, pady=10)

        self.bind("<Return>", lambda event: submit_button.invoke())

        self.textbox1 = tk.Text(self, width=70, height=20)
        self.textbox1.grid(row=12, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)

        self.textbox2 = tk.Text(self, width=20, height=10)
        self.textbox2.grid(row=12, column=8, columnspan=9, padx=10, pady=10, sticky=tk.NSEW)

        scrollbar = tk.Scrollbar(self, command=self.textbox1.yview)
        scrollbar.grid(row=12, column=20, rowspan=3, sticky=tk.NS)
        self.textbox1.config(yscrollcommand=scrollbar.set)

        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]

        locale.setlocale(locale.LC_ALL, '')
        # Create the dataframe
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), 
                               round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6),
                               round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" if isinstance(x, int) or isinstance(x,
                                                                                                                float) else x)

        # Create Treeview
        self.treeview = ttk.Treeview(self, columns=list(df.columns), show='headings')
        for col in df.columns:
            self.treeview.heading(col, text=col, anchor='w')  # Set anchor to 'w' (west) for left alignment
            self.treeview.column(col, width=120, anchor='w')

        for _, row in df.iterrows():
            self.treeview.insert('', 'end', values=list(row))

        self.treeview.grid(row=12, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)

        self.formatted_df = self.format_dataframe()
        # Scrollbar for Treeview
        visualization_button = tk.Button(self, text="Visualization",
                                          command=self.show_visualizations, width=10)
        visualization_button.grid(row=11, column=1, pady=10)

    def format_dataframe(self):
        # Create the dataframe as you did before
        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6),
                                round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6), 
                              round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" if isinstance(x, int) or isinstance(x,
                                                                                                                float) else x)
        return df

    def show_visualizations(self):
        visualization_window = tk.Toplevel(self)
        visualization_window.title("Visualizations")

        formatted_df = self.formatted_df

        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

        # HLA Exposures Pie Chart
        hla_frame = tk.Frame(visualization_window)
        hla_frame.pack(padx=10, pady=10)

        hla_labels = formatted_df['<HLAs>']
        exposure_hla = formatted_df['<Exposure HLA>']
        sizes_hla = [float(size.replace("$", "").replace("m", "").replace(",", "")) if size else 0.0 for size in
                     exposure_hla]

        hla_pie = plt.figure(figsize=(6, 6))
        plt.pie(sizes_hla, labels=hla_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("HLA Exposures Pie Chart")
        plt.tight_layout()

        hla_canvas = FigureCanvasTkAgg(hla_pie, master=hla_frame)
        hla_canvas.draw()
        hla_canvas.get_tk_widget().pack()

        # LA Exposures Pie Chart
        la_frame = tk.Frame(visualization_window)
        la_frame.pack(padx=10, pady=10)

        la_labels = formatted_df['<LAs>']
        exposure_la = formatted_df['<Exposure LA>']
        sizes_la = [float(size.replace("$", "").replace("m", "").replace(",", "")) if size else 0.0 for size in
                    exposure_la]

        la_pie = plt.figure(figsize=(6, 6))
        plt.pie(sizes_la, labels=la_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("LA Exposures Pie Chart")
        plt.tight_layout()

        la_canvas = FigureCanvasTkAgg(la_pie, master=la_frame)
        la_canvas.draw()
        la_canvas.get_tk_widget().pack()

        # IA Exposures Pie Chart
        ia_frame = tk.Frame(visualization_window)
        ia_frame.pack(padx=10, pady=10)

        ia_labels = formatted_df['<IAs>']
        exposure_ia = formatted_df['<Exposure IA>']
        sizes_ia = [float(size.replace("$", "").replace("m", "").replace(",", "")) if size else 0.0 for size in
                    exposure_ia]

        ia_pie = plt.figure(figsize=(6, 6))
        plt.pie(sizes_ia, labels=ia_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("IA Exposures Pie Chart")
        plt.tight_layout()

        ia_canvas = FigureCanvasTkAgg(ia_pie, master=ia_frame)
        ia_canvas.draw()
        ia_canvas.get_tk_widget().pack()

        # DN Exposures Pie Chart
        dn_frame = tk.Frame(visualization_window)
        dn_frame.pack(padx=10, pady=10)

        dn_labels = formatted_df['<Derivative Notionals>']
        exposure_dn = formatted_df['<Exposure DN>']
        sizes_dn = [float(size.replace("$", "").replace("m", "").replace(",", "")) if size else 0.0 for size in
                    exposure_dn]

        dn_pie = plt.figure(figsize=(6, 6))
        plt.pie(sizes_dn, labels=dn_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("DN Exposures Pie Chart")
        plt.tight_layout()

        dn_canvas = FigureCanvasTkAgg(dn_pie, master=dn_frame)
        dn_canvas.draw()
        dn_canvas.get_tk_widget().pack()

    def submit_inputs(self):
        # Add your code here to handle the submission and update the text boxes
        lcr5, lcr10, lcr30, lcr90, lcr365, lcr5_c, lcr10_c, lcr30_c, lcr90_c, lcr365_c = self.reallocations()

        self.textbox2.delete(1.0, tk.END)
        self.textbox2.insert(tk.END, f"Date: {self.expected_date_str}\n\n")

        self.textbox2.insert(tk.END, "*LCRs (before change | after change)*\n")
        self.textbox2.insert(tk.END,
                             f"LCR 5: {locale.format_string('%.2f', lcr5)} | {locale.format_string('%.2f', lcr5_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 10: {locale.format_string('%.2f', lcr10)} | {locale.format_string('%.2f', lcr10_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 30: {locale.format_string('%.2f', lcr30)} | {locale.format_string('%.2f', lcr30_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 90: {locale.format_string('%.2f', lcr90)} | {locale.format_string('%.2f', lcr90_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 365: {locale.format_string('%.2f', lcr365)} | {locale.format_string('%.2f', lcr365_c)}\n\n")

    def reallocations(self):
        df = self.data_handler.dataframe
        df_filtered = df.loc[:, self.expected_date_str]

        # HLAs
        cash = df_filtered[25]
        demandDeposit = df_filtered[26]
        redeemableGIC = df_filtered[27]
        short_term_money_market = df_filtered[28]

        # LAs
        repo = df_filtered[30]
        termDeposit_1d_1m = df_filtered[31]
        non_redeemableGIC_1d_1m = df_filtered[32]
        termDeposit_1m_3m = df_filtered[33]
        non_redeemableGIC_1m_3m = df_filtered[34]

        passiveEQ = df_filtered[35]
        passiveFIN = df_filtered[36]
        passiveFILD = df_filtered[37]
        passiveIGC = df_filtered[38]

        # IAs
        activeARS = df_filtered[40]
        activeEQ = df_filtered[41]
        activeFIN = df_filtered[42]
        activeFILD = df_filtered[43]
        corporateCredit = df_filtered[44]

        # get SLRs
        slr_5d = df_filtered[45]
        slr_10d = df_filtered[46]
        slr_30d = df_filtered[47]
        slr_90d = df_filtered[48]
        slr_365d = df_filtered[49]

        # cashflows
        ci_5 = df_filtered[61]
        ci_10 = df_filtered[62]
        ci_30 = df_filtered[63]
        ci_90 = df_filtered[64]
        ci_365 = df_filtered[65]

        co_5 = df_filtered[66]
        co_10 = df_filtered[67]
        co_30 = df_filtered[68]
        co_90 = df_filtered[69]
        co_365 = df_filtered[70]

        # derivative notionals
        currency_hedge = df_filtered[51]
        EQ_hedge = df_filtered[52]
        FIN_hedge = df_filtered[53]
        FILD_hedge = df_filtered[54]
        IGC_hedge = df_filtered[55]

        ################### get changes in $ #####################
        # HLA
        cash_c = self.cash.get() + cash
        demandDeposit_c = self.demandDeposit.get() + demandDeposit
        redeemableGIC_c = self.redeemableGIC.get() + redeemableGIC
        short_term_money_market_c = self.short_term_money_market.get() + short_term_money_market

        # LA
        repo_c = self.repo.get() + repo
        termDeposit_1d_1m_c = self.termDeposit_1d_1m.get() + termDeposit_1d_1m
        non_redeemableGIC_1d_1m_c = self.non_redeemableGIC_1d_1m.get() + non_redeemableGIC_1d_1m
        termDeposit_1m_3m_c = self.termDeposit_1m_3m.get() + termDeposit_1m_3m
        non_redeemableGIC_1m_3m_c = self.non_redeemableGIC_1m_3m.get() + non_redeemableGIC_1m_3m
        passiveEQ_c = self.passiveEQ.get() + passiveEQ
        passiveFIN_c = self.passiveFIN.get() + passiveFIN  # DN -FIN
        passiveFILD_c = self.passiveFILD.get() + passiveFILD  # DN -FILD
        passiveIGC_c = self.passiveIGC.get() + passiveIGC

        # IA
        activeARS_c = self.activeARS.get() + activeARS
        activeEQ_c = self.activeEQ.get() + activeEQ
        activeFIN_c = self.activeFIN.get() + activeFIN  # DN -FIN
        activeFILD_c = self.activeFILD.get() + activeFILD  # DN -FILD
        corporateCredit_c = self.corporateCredit.get() + corporateCredit  # DN -IGC

        # privates
        activePrivateEQ_c = self.activePrivateEQ.get()
        activeREITs_c = self.activeREITs.get()  # DN -C
        infra_c = self.infra.get()  # DN -C
        privateDebt_c = self.privateDebt.get()  # DN -C

        #################### Cash Asset Aggregates A4 Shock Recalculations#################

        HLA = cash_c + demandDeposit_c + redeemableGIC_c + short_term_money_market_c

        la_30_c = (repo_c + termDeposit_1d_1m_c + non_redeemableGIC_1d_1m_c) + (passiveEQ_c * 0.85
                    ) + (passiveFIN_c * 0.95) + \
                  (passiveFILD_c * 0.9) + (passiveIGC_c * 0.85)

        la_90_c = (repo_c + termDeposit_1d_1m_c + non_redeemableGIC_1d_1m_c + termDeposit_1m_3m_c + \
                   non_redeemableGIC_1m_3m_c) + (passiveEQ_c * 0.75) + (passiveFIN_c * 0.95) + \
                  (passiveFILD_c * 0.85) + (passiveIGC_c * 0.75)

        la_365_c = (repo_c + termDeposit_1d_1m_c + non_redeemableGIC_1d_1m_c + termDeposit_1m_3m_c + \
                    non_redeemableGIC_1m_3m_c) + (passiveEQ_c * 0.6) + (passiveFIN_c * 0.9) + \
                   (passiveFILD_c * 0.75) + (passiveIGC_c * 0.6)

        ia_365_c = (activeARS_c * 0.6 * 0.96) + (activeEQ_c * 0.6 * 0.96) + (activeFIN_c * 0.9 * 0.97) + \
                   (activeFILD_c * 0.75 * 0.97) + (corporateCredit_c * 0.6 * 0.96)

        ################ Derivative Notional & SLR Changes Recalculations #############
        FIN_hedge_c = FIN_hedge + self.DN_fin.get()
        FILD_hedge_c = FILD_hedge + self.DN_fild.get()
        IGC_hedge_c = IGC_hedge + self.DN_igc.get()
        currency_hedge_c = currency_hedge + self.DN_currencies.get()
        EQ_hedge_c = EQ_hedge + self.DN_equities.get()

        slr_5d_c = currency_hedge_c * 0.05 + EQ_hedge_c * 0.15 + \
                   FIN_hedge_c * 0.05 + FILD_hedge_c * 0.1 + IGC_hedge_c * 0.15

        slr_10d_c = currency_hedge_c * 0.05 + EQ_hedge_c * 0.15 + \
                    FIN_hedge_c * 0.05 + FILD_hedge_c * 0.1 + IGC_hedge_c * 0.15

        slr_30d_c = currency_hedge_c * 0.05 + EQ_hedge_c * 0.15 + \
                    FIN_hedge_c * 0.05 + FILD_hedge_c * 0.1 + IGC_hedge_c * 0.15

        slr_90d_c = currency_hedge_c * 0.1 + EQ_hedge_c * 0.25 + \
                    FIN_hedge_c * 0.05 + FILD_hedge_c * 0.15 + IGC_hedge_c * 0.25

        slr_365d_c = currency_hedge_c * 0.2 + EQ_hedge_c * 0.4 + \
                     FIN_hedge_c * 0.1 + FILD_hedge_c * 0.25 + IGC_hedge_c * 0.4

        ############### Recalculate LCRs ###############
        lcr5_c = (HLA + ci_5) / (slr_5d_c - co_5)
        lcr10_c = (HLA + ci_5 + ci_10) / (slr_10d_c - (co_5 + co_10))
        lcr30_c = (HLA + la_30_c + ci_5 + ci_10 + ci_30) / (slr_30d_c - (co_5 + co_10) - (co_30) * 1.2)
        lcr90_c = (HLA + la_90_c + ci_5 + ci_10 + ci_30 + ci_90) / (slr_90d_c -
                                                                    (co_5 + co_10) - (co_30 + co_90) * 1.2)
        lcr365_c = (HLA + la_365_c + ia_365_c + ci_5 + ci_10 + ci_30 + ci_90 \
                    + ci_365) / (slr_365d_c - (co_5 + co_10) - (co_30 + co_90 + co_365) * 1.2)
        # original LCRs
        lcr5 = df_filtered[0]
        lcr10 = df_filtered[1]
        lcr30 = df_filtered[2]
        lcr90 = df_filtered[3]
        lcr365 = df_filtered[4]

        return lcr5, lcr10, lcr30, lcr90, lcr365, lcr5_c, lcr10_c, lcr30_c, lcr90_c, lcr365_c


#rename replacement?
class Shock_Haircut_Change(tk.Toplevel):
    def __init__(self, parent, expected_date_str, data_handler):
        tk.Toplevel.__init__(self, parent)
        self.expected_date_str = expected_date_str
        self.data_handler = data_handler
        self.geometry("1550x680")
        label = tk.Label(self, text="Framework Shock/Haircut Change", 
                         font = ("Arial", 16))
        label.grid(padx=10, pady=10)

        #shock 0-5 days
        self.equities_0_5 = tk.IntVar()
        self.fin_0_5 = tk.IntVar()
        self.fild_0_5 = tk.IntVar()
        self.ars_0_5 = tk.IntVar()
        self.igc_0_5 = tk.IntVar()
        self.currency_0_5 = tk.IntVar()

        #shock 6-10 days
        self.equities_6_10 = tk.IntVar()
        self.fin_6_10 = tk.IntVar()
        self.fild_6_10 = tk.IntVar()
        self.ars_6_10 = tk.IntVar()
        self.igc_6_10 = tk.IntVar()
        self.currency_6_10 = tk.IntVar()

        #shock 11-30 days
        self.equities_11_30 = tk.IntVar()
        self.fin_11_30 = tk.IntVar()
        self.fild_11_30 = tk.IntVar()
        self.ars_11_30 = tk.IntVar()
        self.igc_11_30 = tk.IntVar()
        self.currency_11_30 = tk.IntVar()

        #shock 31-90 days
        self.equities_31_90 = tk.IntVar()
        self.fin_31_90 = tk.IntVar()
        self.fild_31_90 = tk.IntVar()
        self.ars_31_90 = tk.IntVar()
        self.igc_31_90 = tk.IntVar()
        self.currency_31_90 = tk.IntVar()

        #shock 91-365 days
        self.equities_1y = tk.IntVar()
        self.fin_1y = tk.IntVar()
        self.fild_1y = tk.IntVar()
        self.ars_1y = tk.IntVar()
        self.igc_1y = tk.IntVar()
        self.currency_1y = tk.IntVar()

        #haircuts
        self.equities_hc = tk.IntVar()
        self.fin_hc = tk.IntVar()
        self.fild_hc = tk.IntVar()
        self.ars_hc = tk.IntVar()
        self.igc_hc = tk.IntVar()
        self.currency_hc = tk.IntVar()


        #input labels
        adj_label0 = tk.Label(self, text="Shock/Haircut Change", font=("Helvetica", 16, "bold"))
        adj_label0.grid(row=1, column=0, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label1 = tk.Label(self, text="Shock % 5D",
                              font=("Arial", 12, "bold"))
        adj_label1.grid(row=1, column=1, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label2 = tk.Label(self, text="Shock % 10D",
                              font=("Arial", 12, "bold"))
        adj_label2.grid(row=1, column=2, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label3 = tk.Label(self, text="Shock % 30D",
                              font=("Arial", 12, "bold"))
        adj_label3.grid(row=1, column=3, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label4 = tk.Label(self, text="Shock % 90D",
                              font=("Arial", 12, "bold"))
        adj_label4.grid(row=1, column=4, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label5 = tk.Label(self, text="Shock % 365D",
                              font=("Arial", 12, "bold"))
        adj_label5.grid(row=1, column=5, pady=5, padx=10, sticky=tk.W, columnspan=4)

        adj_label6 = tk.Label(self, text="Haircut %",
                              font=("Arial", 12, "bold"))
        adj_label6.grid(row=1, column=6, pady=5, padx=10, sticky=tk.W, columnspan=4)


        #input fields
        inp_lb1 = tk.Label(self, text="Equities")
        inp_lb1.grid(row=2, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent1_1 = tk.Entry(self, textvariable=self.equities_0_5)
        inp_ent1_1.grid(row=2, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent1_2 = tk.Entry(self, textvariable=self.equities_6_10)
        inp_ent1_2.grid(row=2, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent1_3 = tk.Entry(self, textvariable=self.equities_11_30)
        inp_ent1_3.grid(row=2, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent1_4 = tk.Entry(self, textvariable=self.equities_31_90)
        inp_ent1_4.grid(row=2, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent1_5 = tk.Entry(self, textvariable=self.equities_1y)
        inp_ent1_5.grid(row=2, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent1_6 = tk.Entry(self, textvariable=self.equities_hc)
        inp_ent1_6.grid(row=2, column=6, pady=5, padx=10, sticky=tk.W)


        inp_lb2 = tk.Label(self, text="FI Nominals")
        inp_lb2.grid(row=3, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent2_1 = tk.Entry(self, textvariable=self.fin_0_5)
        inp_ent2_1.grid(row=3, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent2_2 = tk.Entry(self, textvariable=self.fin_6_10)
        inp_ent2_2.grid(row=3, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent2_3 = tk.Entry(self, textvariable=self.fin_11_30)
        inp_ent2_3.grid(row=3, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent2_4 = tk.Entry(self, textvariable=self.fin_31_90)
        inp_ent2_4.grid(row=3, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent2_5 = tk.Entry(self, textvariable=self.fin_1y)
        inp_ent2_5.grid(row=3, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent2_6 = tk.Entry(self, textvariable=self.fin_hc)
        inp_ent2_6.grid(row=3, column=6, pady=5, padx=10, sticky=tk.W)


        inp_lb3 = tk.Label(self, text="FI Long Durational")
        inp_lb3.grid(row=4, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent3_1 = tk.Entry(self, textvariable=self.fild_0_5)
        inp_ent3_1.grid(row=4, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent3_2 = tk.Entry(self, textvariable=self.fild_6_10)
        inp_ent3_2.grid(row=4, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent3_3 = tk.Entry(self, textvariable=self.fild_11_30)
        inp_ent3_3.grid(row=4, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent3_4 = tk.Entry(self, textvariable=self.fild_31_90)
        inp_ent3_4.grid(row=4, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent3_5 = tk.Entry(self, textvariable=self.fild_1y)
        inp_ent3_5.grid(row=4, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent3_6 = tk.Entry(self, textvariable=self.fild_hc)
        inp_ent3_6.grid(row=4, column=6, pady=5, padx=10, sticky=tk.W)


        inp_lb4 = tk.Label(self, text="ARS")
        inp_lb4.grid(row=5, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent4_1 = tk.Entry(self, textvariable=self.ars_0_5)
        inp_ent4_1.grid(row=5, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent4_2 = tk.Entry(self, textvariable=self.ars_6_10)
        inp_ent4_2.grid(row=5, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent4_3 = tk.Entry(self, textvariable=self.ars_11_30)
        inp_ent4_3.grid(row=5, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent4_4 = tk.Entry(self, textvariable=self.ars_31_90)
        inp_ent4_4.grid(row=5, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent4_5 = tk.Entry(self, textvariable=self.ars_1y)
        inp_ent4_5.grid(row=5, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent4_6 = tk.Entry(self, textvariable=self.ars_hc)
        inp_ent4_6.grid(row=5, column=6, pady=5, padx=10, sticky=tk.W)


        inp_lb5 = tk.Label(self, text="Investment Grade Credits")
        inp_lb5.grid(row=6, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent5_1 = tk.Entry(self, textvariable=self.igc_0_5)
        inp_ent5_1.grid(row=6, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent5_2 = tk.Entry(self, textvariable=self.igc_6_10)
        inp_ent5_2.grid(row=6, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent5_3 = tk.Entry(self, textvariable=self.igc_11_30)
        inp_ent5_3.grid(row=6, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent5_4 = tk.Entry(self, textvariable=self.igc_31_90)
        inp_ent5_4.grid(row=6, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent5_5 = tk.Entry(self, textvariable=self.igc_1y)
        inp_ent5_5.grid(row=6, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent5_6 = tk.Entry(self, textvariable=self.igc_hc)
        inp_ent5_6.grid(row=6, column=6, pady=5, padx=10, sticky=tk.W)


        inp_lb6 = tk.Label(self, text="Currency Derivatives")
        inp_lb6.grid(row=7, column=0, pady=5, padx=10, sticky=tk.W)
        
        inp_ent6_1 = tk.Entry(self, textvariable=self.currency_0_5)
        inp_ent6_1.grid(row=7, column=1, pady=5, padx=10, sticky=tk.W)
        inp_ent6_2 = tk.Entry(self, textvariable=self.currency_6_10)
        inp_ent6_2.grid(row=7, column=2, pady=5, padx=10, sticky=tk.W)
        inp_ent6_3 = tk.Entry(self, textvariable=self.currency_11_30)
        inp_ent6_3.grid(row=7, column=3, pady=5, padx=10, sticky=tk.W)
        inp_ent6_4 = tk.Entry(self, textvariable=self.currency_31_90)
        inp_ent6_4.grid(row=7, column=4, pady=5, padx=10, sticky=tk.W)
        inp_ent6_5 = tk.Entry(self, textvariable=self.currency_1y)
        inp_ent6_5.grid(row=7, column=5, pady=5, padx=10, sticky=tk.W)
        inp_ent6_6 = tk.Entry(self, textvariable=self.currency_hc)
        inp_ent6_6.grid(row=7, column=6, pady=5, padx=10, sticky=tk.W)


        #additional formatting
        submit_button = tk.Button(self, text="Submit", command=self.submit_inputs)
        submit_button.grid(row=8, column=0, pady=10, columnspan=12)
        self.bind("<Return>", lambda event: submit_button.invoke())

        self.textbox1 = tk.Text(self, width=70, height=20)
        self.textbox1.grid(row=12, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)

        self.textbox2 = tk.Text(self, width=37, height=10)
        self.textbox2.grid(row=12, column=8, columnspan=13, padx=10, pady=10, sticky=tk.NSEW)

        scrollbar = tk.Scrollbar(self, command=self.textbox1.yview)
        scrollbar.grid(row=12, column=22, rowspan=3, sticky=tk.NS)
        self.textbox1.config(yscrollcommand=scrollbar.set)


        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]

        locale.setlocale(locale.LC_ALL, '')
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), 
                               round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6),
                               round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" 
            if isinstance(x, int) or isinstance(x, float) else x
        )

        self.treeview = ttk.Treeview(self, columns=list(df.columns), show='headings')
        for col in df.columns:
            self.treeview.heading(col, text=col, anchor='w')  # Set anchor to 'w' (west) for left alignment
            self.treeview.column(col, width=120, anchor='w')

        for _, row in df.iterrows():
            self.treeview.insert('', 'end', values=list(row))

        self.treeview.grid(row=12, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)
        self.formatted_df = self.format_dataframe()

        # df_str = df.to_string(index=False, col_space=15, justify='left')
        # font = tkfont.Font(family='Courier New', size=10)
        # aligned_lines = [line.ljust(120) for line in df_str.split('\n')]
        # df_aligned_str = '\n'.join(aligned_lines)

        # self.textbox1.insert(tk.END, df_aligned_str)
        # self.textbox1.configure(font=font)

    def format_dataframe(self):
        # Create the dataframe as you did before
        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), 
                               round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6), 
                              round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" if isinstance(x, int) or isinstance(x,
                                                                                                                float) else x)
        return df
    
    def submit_inputs(self):
        lcr5, lcr10, lcr30, lcr90, lcr365, lcr5_c, lcr10_c, lcr30_c, lcr90_c, \
            lcr365_c = self.shock_hc()

        self.textbox2.delete(1.0, tk.END)
        self.textbox2.insert(tk.END, f"Date: {self.expected_date_str}\n\n")

        self.textbox2.insert(tk.END, "*LCRs (before change | after change)*\n")
        self.textbox2.insert(tk.END,
                             f"LCR 5: {locale.format_string('%.2f', lcr5)} | {locale.format_string('%.2f', lcr5_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 10: {locale.format_string('%.2f', lcr10)} | {locale.format_string('%.2f', lcr10_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 30: {locale.format_string('%.2f', lcr30)} | {locale.format_string('%.2f', lcr30_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 90: {locale.format_string('%.2f', lcr90)} | {locale.format_string('%.2f', lcr90_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 365: {locale.format_string('%.2f', lcr365)} | {locale.format_string('%.2f', lcr365_c)}\n\n")


    def shock_hc(self):
        df = self.data_handler.dataframe
        df_filtered = df.loc[:, self.expected_date_str]

        # HLAs
        cash = df_filtered[25]
        demandDeposit = df_filtered[26]
        redeemableGIC = df_filtered[27]
        short_term_money_market = df_filtered[28]

        # LAs
        repo = df_filtered[30]
        termDeposit_1d_1m = df_filtered[31]
        non_redeemableGIC_1d_1m = df_filtered[32]
        termDeposit_1m_3m = df_filtered[33]
        non_redeemableGIC_1m_3m = df_filtered[34]
        passiveEQ = df_filtered[35]
        passiveFIN = df_filtered[36]
        passiveFILD = df_filtered[37]
        passiveIGC = df_filtered[38]

        # IAs
        activeARS = df_filtered[40]
        activeEQ = df_filtered[41]
        activeFIN = df_filtered[42]
        activeFILD = df_filtered[43]
        corporateCredit = df_filtered[44]

        # cashflows
        ci_5 = df_filtered[61]
        ci_10 = df_filtered[62]
        ci_30 = df_filtered[63]
        ci_90 = df_filtered[64]
        ci_365 = df_filtered[65]

        co_5 = df_filtered[66]
        co_10 = df_filtered[67]
        co_30 = df_filtered[68]
        co_90 = df_filtered[69]
        co_365 = df_filtered[70]

        # derivative notionals
        EQ_hedge = df_filtered[52]
        FIN_hedge = df_filtered[53]
        FILD_hedge = df_filtered[54]
        IGC_hedge = df_filtered[55]
        currency_hedge = df_filtered[51]

        HLA = cash + demandDeposit + redeemableGIC + short_term_money_market

        #new shock & hc % (ignore all % in framework from here.)
        eq_sh_5 = self.equities_0_5.get()/100
        eq_sh_10 = self.equities_6_10.get()/100
        eq_sh_30 = self.equities_11_30.get()/100
        eq_sh_90 = self.equities_31_90.get()/100
        eq_sh_1y = self.equities_1y.get()/100
        eq_hc = self.equities_hc.get()/100

        fin_sh_5 = self.fin_0_5.get()/100
        fin_sh_10 = self.fin_6_10.get()/100
        fin_sh_30 = self.fin_11_30.get()/100
        fin_sh_90 = self.fin_31_90.get()/100
        fin_sh_1y = self.fin_1y.get()/100
        fin_hc = self.fin_hc.get()/100

        fild_sh_5 = self.fild_0_5.get()/100
        fild_sh_10 = self.fild_6_10.get()/100
        fild_sh_30 = self.fild_11_30.get()/100
        fild_sh_90 = self.fild_31_90.get()/100
        fild_sh_1y = self.fild_1y.get()/100
        fild_hc = self.fild_hc.get()/100

        ars_sh_5 = self.ars_0_5.get()/100
        ars_sh_10 = self.ars_6_10.get()/100
        ars_sh_30 = self.ars_11_30.get()/100
        ars_sh_90 = self.ars_31_90.get()/100
        ars_sh_1y = self.ars_1y.get()/100
        ars_hc = self.ars_hc.get()/100

        igc_sh_5 = self.igc_0_5.get()/100
        igc_sh_10 = self.igc_6_10.get()/100
        igc_sh_30 = self.igc_11_30.get()/100
        igc_sh_90 = self.igc_31_90.get()/100
        igc_sh_1y = self.igc_1y.get()/100
        igc_hc = self.igc_hc.get()/100

        cur_sh_5 = self.currency_0_5.get()/100
        cur_sh_10 = self.currency_6_10.get()/100
        cur_sh_30 = self.currency_11_30.get()/100
        cur_sh_90 = self.currency_31_90.get()/100
        cur_sh_1y = self.currency_1y.get()/100
        cur_hc = self.currency_hc.get()/100

        #changes all across 5 time horizons (physical & SLRs)
        la_updated_30 = (
            repo + termDeposit_1d_1m + non_redeemableGIC_1d_1m + 
            (passiveEQ * (1-eq_sh_30)) + (passiveFIN * (1-fin_sh_30)) +
              (passiveFILD * (1-fild_sh_30)) + (passiveIGC * (1-igc_sh_30))
        )

        la_updated_90 = (
            repo + termDeposit_1d_1m + non_redeemableGIC_1d_1m + termDeposit_1m_3m + 
            non_redeemableGIC_1m_3m +
            (passiveEQ * (1-eq_sh_90)) + (passiveFIN * (1-fin_sh_90)) + (passiveFILD * 
                                                                         (1-fild_sh_90)) +
            (passiveIGC * (1-igc_sh_90))
        )

        la_updated_1y = (
            repo + termDeposit_1d_1m + non_redeemableGIC_1d_1m + termDeposit_1m_3m + 
            non_redeemableGIC_1m_3m +
            (passiveEQ * (1-eq_sh_1y)) + (passiveFIN * (1-fin_sh_1y)) + 
            (passiveFILD * (1-fild_sh_1y)) + (passiveIGC * (1-igc_sh_1y))
        )

        ia_updated_1y = (
            (activeEQ * (1-eq_sh_1y) * (1-eq_hc)) + (activeARS * (1-ars_sh_1y) * (1-ars_hc)) +
            (activeFIN * (1-fin_sh_1y) * (1-fin_hc)) + (activeFILD * (1-fild_sh_1y) *
                                                         (1-fild_hc)) + 
            (corporateCredit * (1-igc_sh_1y) * (1-igc_hc))
        )

        slr_updated_5d = EQ_hedge * eq_sh_5 + FIN_hedge * fin_sh_5 + FILD_hedge * fild_sh_5 + IGC_hedge * igc_sh_5 + currency_hedge * cur_sh_5
        slr_updated_10d = EQ_hedge * eq_sh_10 + FIN_hedge * fin_sh_10 + FILD_hedge * fild_sh_10 + IGC_hedge * igc_sh_10 + currency_hedge * cur_sh_10
        slr_updated_30d = EQ_hedge * eq_sh_30 + FIN_hedge * fin_sh_30 + FILD_hedge * fild_sh_30 + IGC_hedge * igc_sh_30 + currency_hedge * cur_sh_30
        slr_updated_90d = EQ_hedge * eq_sh_90 + FIN_hedge * fin_sh_90 + FILD_hedge * fild_sh_90 + IGC_hedge * igc_sh_90 + currency_hedge * cur_sh_90
        slr_updated_1y = EQ_hedge * eq_sh_1y + FIN_hedge * fin_sh_1y + FILD_hedge * fild_sh_1y + IGC_hedge * igc_sh_1y + currency_hedge * cur_sh_1y


        #LCR calculation
        lcr5c = (HLA + ci_5) / (slr_updated_5d - co_5)
        lcr10c = (HLA + ci_5 + ci_10) / (slr_updated_10d - (co_5 + co_10))
        lcr30c = (HLA + la_updated_30 + ci_5 + ci_10 + ci_30) / (
            slr_updated_30d - (co_5 + co_10)- co_30*1.2
        )
        lcr90c = (HLA + la_updated_90 + ci_5 + ci_10 + ci_30 + ci_90)/(
            slr_updated_90d - (co_5 + co_10)- (co_30 + co_90)*1.2
        )
        lcr365c = (HLA + la_updated_1y + ia_updated_1y + ci_5 + ci_10 + ci_30 + ci_90 + ci_365)/(
            slr_updated_1y - (co_5 + co_10)- (co_30 + co_90 + co_365)*1.2
        )

        # original LCRs
        lcr5 = df_filtered[0]
        lcr10 = df_filtered[1]
        lcr30 = df_filtered[2]
        lcr90 = df_filtered[3]
        lcr365 = df_filtered[4]

        return lcr5, lcr10, lcr30, lcr90, lcr365, lcr5c, lcr10c, lcr30c, lcr90c, lcr365c
        
    

class MarketShock(tk.Toplevel):
    def __init__(self, parent, expected_date_str, data_handler):
        tk.Toplevel.__init__(self, parent)
        self.title("Market Shock")
        self.expected_date_str = expected_date_str
        self.data_handler = data_handler
        self.geometry("850x975")
        label = tk.Label(self, text="Instant Market Shock Simulations", font=("Arial", 16))
        label.pack(pady=10, padx=10)

        self.equities = tk.IntVar()
        self.fixedIncomeNominal = tk.IntVar()
        self.fixedIncomeLongDurational = tk.IntVar()
        self.currencies = tk.IntVar()
        self.investmentGradeCredit = tk.IntVar()
        self.absoluteReturnStrategies = tk.IntVar()

        input_label1 = tk.Label(self, text="Enter market shock % for equities (+/-)")
        input_label1.pack(pady=5, padx=10, anchor=tk.W)
        input_entry1 = tk.Entry(self, textvariable=self.equities)
        input_entry1.pack(pady=5, padx=10, anchor=tk.W)

        input_label2 = tk.Label(self, text="Enter market shock % for fixed income nominals (+/-)")
        input_label2.pack(pady=5, padx=10, anchor=tk.W)
        input_entry2 = tk.Entry(self, textvariable=self.fixedIncomeNominal)
        input_entry2.pack(pady=5, padx=10, anchor=tk.W)

        input_label3 = tk.Label(self, text="Enter market shock % for fixed income long durationals (+/-)")
        input_label3.pack(pady=5, padx=10, anchor=tk.W)
        input_entry3 = tk.Entry(self, textvariable=self.fixedIncomeLongDurational)
        input_entry3.pack(pady=5, padx=10, anchor=tk.W)

        input_label5 = tk.Label(self, text="Enter market shock % for investment grade credits (+/-)")
        input_label5.pack(pady=5, padx=10, anchor=tk.W)
        input_entry5 = tk.Entry(self, textvariable=self.investmentGradeCredit)
        input_entry5.pack(pady=5, padx=10, anchor=tk.W)

        input_label6 = tk.Label(self, text="Enter market shock % for ARS (+/-)")
        input_label6.pack(pady=5, padx=10, anchor=tk.W)
        input_entry6 = tk.Entry(self, textvariable=self.absoluteReturnStrategies)
        input_entry6.pack(pady=5, padx=10, anchor=tk.W)

        input_label4 = tk.Label(self,
                                text="Enter market shock % for currency derivatives (+/- CAD-appreciation/depreciation against Non-CAD)")
        input_label4.pack(pady=5, padx=10, anchor=tk.W)
        input_entry4 = tk.Entry(self, textvariable=self.currencies)
        input_entry4.pack(pady=5, padx=10, anchor=tk.W)

        submit_button = tk.Button(self, text="Submit", command=self.submit_inputs, width=10)
        submit_button.pack(pady=10)
        self.bind("<Return>", lambda event: submit_button.invoke())

        self.textbox = tk.Text(self, width=30, height=35)
        self.textbox.pack(fill=tk.BOTH, padx=10, pady=10)

        scrollbar = tk.Scrollbar(self, command=self.textbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        comment_label = Label(self, text="* Currency shock is only for derivatives *", fg="red")
        comment_label.pack(anchor=W, padx=10, pady=10)

    def submit_inputs(self):
        lcr5, lcr10, lcr30, lcr90, lcr365, new_lcr_5, new_lcr_10, new_lcr_30, new_lcr_90, new_lcr_365, \
            eq_initial, fin_initial, fild_initial, igc_initial, ars_initial, \
            EQ_hedge, FIN_hedge, FILD_hedge, IGC_hedge, currency_hedge, \
            alpha_MTM_c_in_eq, alpha_MTM_c_out_eq, alpha_MTM_c_in_fin, alpha_MTM_c_out_fin, \
            alpha_MTM_in_FILD, alpha_MTM_out_FILD, alpha_MTM_in_FX, alpha_MTM_out_FX, \
            alpha_MTM_in_igc, alpha_MTM_out_igc, MTM_eq, MTM_fin, MTM_fild, MTM_fx, MTM_igc, \
            equities, FIN, FILD, IGC, ARS = self.market_shock()

        locale.setlocale(locale.LC_ALL, '')  # Set the locale to use commas as thousand separators

        # date
        self.textbox.insert(tk.END, f"Date: {self.expected_date_str}\n\n")

        # parameters
        self.textbox.insert(tk.END, "**Inputs**\n")
        self.textbox.insert(tk.END, f"Equities %: {locale.format_string('%.0f', self.equities.get(), grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Fixed Income Nominal %: {locale.format_string('%.0f', self.fixedIncomeNominal.get(), grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Fixed Income LD %: {locale.format_string('%.0f', self.fixedIncomeLongDurational.get(), grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Investment Grade Credit %: {locale.format_string('%.0f', self.investmentGradeCredit.get(), grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"ARS %: {locale.format_string('%.0f', self.absoluteReturnStrategies.get(), grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Currency %: {locale.format_string('%.0f', self.currencies.get(), grouping=True)}\n\n")

        # LCRs
        self.textbox.insert(tk.END, "**LCRs (before shock | after shock)**\n")
        self.textbox.insert(tk.END,
                            f"LCR 5: {locale.format_string('%.2f', lcr5)} | {locale.format_string('%.2f', new_lcr_5)}\n")
        self.textbox.insert(tk.END,
                            f"LCR 10: {locale.format_string('%.2f', lcr10)} | {locale.format_string('%.2f', new_lcr_10)}\n")
        self.textbox.insert(tk.END,
                            f"LCR 30: {locale.format_string('%.2f', lcr30)} | {locale.format_string('%.2f', new_lcr_30)}\n")
        self.textbox.insert(tk.END,
                            f"LCR 90: {locale.format_string('%.2f', lcr90)} | {locale.format_string('%.2f', new_lcr_90)}\n")
        self.textbox.insert(tk.END,
                            f"LCR 365: {locale.format_string('%.2f', lcr365)} | {locale.format_string('%.2f', new_lcr_365)}\n\n")

        # cash assets after shock
        self.textbox.insert(tk.END, "**Cash Assets**\n")
        self.textbox.insert(tk.END,
                            f"Equities before shock: {locale.format_string('%.0f', equities, grouping=True)} | Equities after shock: {locale.format_string('%.0f', eq_initial, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"FI Nominal before shock {locale.format_string('%.0f', FIN, grouping=True)} | FI Nominal after shock: {locale.format_string('%.0f', fin_initial, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"FI Long Durational before shock {locale.format_string('%.0f', FILD, grouping=True)} | FI Long Durational after shock: {locale.format_string('%.0f', fild_initial, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Investment Grade Credit before shock {locale.format_string('%.0f', IGC, grouping=True)} | Investment Grade Credit after shock: {locale.format_string('%.0f', igc_initial, grouping=True)}\n")

        self.textbox.insert(tk.END,
                            f"ARS before shock: {locale.format_string('%.0f', ARS, grouping=True)} | ARS after shock: {locale.format_string('%.0f', ars_initial, grouping=True)}\n\n")

        # derivative notionals
        self.textbox.insert(tk.END, "**Derivative Notional**\n")
        self.textbox.insert(tk.END,
                            f"Derivative Notional Equity: {locale.format_string('%.0f', EQ_hedge, grouping=True)}| Incremental MTM before Buffer: {locale.format_string('%.0f', MTM_eq, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Derivative Notional Fixed Income Nominal: {locale.format_string('%.0f', FIN_hedge, grouping=True)} | Incremental MTM before Buffer: {locale.format_string('%.0f', MTM_fin, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Derivative Notional Fixed Income Long Durational: {locale.format_string('%.0f', FILD_hedge, grouping=True)} | Incremental MTM before Buffer: {locale.format_string('%.0f', MTM_fild, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Derivative Notional Investment Grade Credit: {locale.format_string('%0.f', IGC_hedge, grouping=True)} | Incremental MTM before Buffer: {locale.format_string('%.0f', MTM_igc, grouping=True)}\n")
        self.textbox.insert(tk.END,
                            f"Derivative Notional Currency: {locale.format_string('%.0f', currency_hedge, grouping=True)} | Incremental MTM before Buffer: {locale.format_string('%.0f', MTM_fx, grouping=True)}\n")

        self.textbox.insert(tk.END,
                            "#############################################################################################\n\n\n")

        # Append the results to the data handler
        scenario_number = len(scenario_data) + 1
        scenario = {
            "Scenario": f"Scenario {scenario_number}",
            "Date": self.expected_date_str,
            "Equities %": locale.format_string('%.0f', self.equities.get(), grouping=True),
            "Fixed Income Nominal %": locale.format_string('%.0f', self.fixedIncomeNominal.get(), grouping=True),
            "Fixed Income LD %": locale.format_string('%.0f', self.fixedIncomeLongDurational.get(), grouping=True),
            "Investment Grade Credit %": locale.format_string('%.0f', self.investmentGradeCredit.get(), grouping=True),
            "ARS %": locale.format_string('%.0f', self.absoluteReturnStrategies.get(), grouping=True),
            "Currency %": locale.format_string('%.0f', self.currencies.get(), grouping=True),

            # "LCR 0-5 days pre-shock": locale.format('%.0f', self.lcr5.get(), grouping=True),

            "Equities after shock": locale.format_string('%.0f', eq_initial, grouping=True),
            "Fixed Income Nominal after shock": locale.format_string('%.0f', fin_initial, grouping=True),
            "Fixed Income LD after shock": locale.format_string('%.0f', fild_initial, grouping=True),
            "Investment Grade Credit after shock": locale.format_string('%.0f', igc_initial, grouping=True),
            "ARS after shock": locale.format_string('%.0f', ars_initial, grouping=True),
            "Derivative Notional Currency after shock": locale.format_string('%.0f', currency_hedge, grouping=True),
            "Derivative Notional Fixed Income Nominal after shock": locale.format_string('%.0f', FIN_hedge,
                                                                                         grouping=True),
            "Derivative Notional Fixed Income Long Durational after shock": locale.format_string('%.0f', FILD_hedge,
                                                                                                 grouping=True),
            "Derivative Notional Investment Grade Credit after shock": locale.format_string('%0.f', IGC_hedge,
                                                                                            grouping=True),
            "LCR 5": locale.format_string('%.2f', new_lcr_5, grouping=True),
            "LCR 10": locale.format_string('%.2f', new_lcr_10, grouping=True),
            "LCR 30": locale.format_string('%.2f', new_lcr_30, grouping=True),
            "LCR 90": locale.format_string('%.2f', new_lcr_90, grouping=True),
            "LCR 365": locale.format_string('%.2f', new_lcr_365, grouping=True)

        }
        scenario_data.append(scenario)

    def market_shock(self):

        ####################################### Phase 2 ##############################################

        # #cash outflows (derivative notionals); phase 2
        # TRS_MTM_outflow_5d = df_filtered[112]
        # FX_MTM_outflow_5d = df_filtered[111]

        # TRS_MTM_outflow_10d = df_filtered[120]
        # FX_MTM_outflow_10d = df_filtered[119]

        # TRS_MTM_outflow_30d = df_filtered[128]
        # FX_MTM_outflow_30d = df_filtered[127]

        # TRS_MTM_outflow_90d = df_filtered[136]
        # FX_MTM_outflow_90d = df_filtered[135]

        # TRS_MTM_outflow_365d = df_filtered[144]
        # FX_MTM_outflow_365d = df_filtered[143]

        # #cash inflows (derivative notionals)
        # TRS_MTM_inflow_5d = df_filtered[76]
        # FX_MTM_inflow_5d = df_filtered[75]

        # TRS_MTM_inflow_10d = df_filtered[83]
        # FX_MTM_inflow_10d = df_filtered[82]

        # TRS_MTM_inflow_30d = df_filtered[90]
        # FX_MTM_inflow_30d = df_filtered[89]

        # TRS_MTM_inflow_90d = df_filtered[97]
        # FX_MTM_inflow_90d = df_filtered[96]

        # TRS_MTM_inflow_365d = df_filtered[104]
        # FX_MTM_inflow_365d = df_filtered[103]
        ##############################################################################################

        # notional should be included & shock & MTM input ()
        # LCR output, columns = params &

        ####################################### Fix ##################################################

        df = self.data_handler.dataframe
        df_filtered = df.loc[:, self.expected_date_str]

        # original LCRs
        # lcr5 = df[0]
        # lcr10 = df[1]
        # lcr30 = df[2]
        # lcr90 = df[3]
        # lcr365 = df[4]

        # cash assets
        equities = df_filtered[35] + df_filtered[41]
        FIN = df_filtered[36] + df_filtered[42]
        FILD = df_filtered[37] + df_filtered[43]
        IGC = df_filtered[44] + df_filtered[38]
        ARS = df_filtered[40]

        # derivative notionals
        EQ_hedge = df_filtered[52]
        FIN_hedge = df_filtered[53]
        FILD_hedge = df_filtered[54]
        IGC_hedge = df_filtered[55]
        currency_hedge = df_filtered[51]

        # get SLRs
        slr_5d = df_filtered[45]
        slr_10d = df_filtered[46]
        slr_30d = df_filtered[47]
        slr_90d = df_filtered[48]
        slr_365d = df_filtered[49]

        # get shock percentages
        eq_shock = self.equities.get() / 100
        fin_shock = self.fixedIncomeNominal.get() / 100
        fild_shock = self.fixedIncomeLongDurational.get() / 100
        igc_shock = self.investmentGradeCredit.get() / 100
        ars_shock = self.absoluteReturnStrategies.get() / 100
        currency_shock = self.currencies.get() / 100

        # effects on cash assets
        eq_initial = equities * (1 + eq_shock)
        fin_initial = FIN * (1 + fin_shock)
        fild_initial = FILD * (1 + fild_shock)
        igc_initial = IGC * (1 + igc_shock)
        ars_initial = ARS * (1 + ars_shock)

        # effects on derivative notionals (MTM)
        EQ_hedge1 = EQ_hedge * (1 + eq_shock)
        FIN_hedge1 = FIN_hedge * (1 + fin_shock)
        FILD_hedge1 = FILD_hedge * (1 + fild_shock)
        IGC_hedge1 = IGC_hedge * (1 + igc_shock)
        currency_hedge1 = currency_hedge * (1 + currency_shock)

        LA_equity_percentage = df_filtered[35] / equities
        IA_equity_percentage = df_filtered[40] / equities
        LA_FIN_perc = df_filtered[36] / FIN
        IA_FIN_perc = df_filtered[41] / FIN
        LA_FILD_perc = df_filtered[37] / FILD
        IA_FILD_perc = df_filtered[42] / FILD

        # LA after shock data
        la30_after_shock = df_filtered[20]
        la90_after_shock = df_filtered[21]
        la365_after_shock = df_filtered[22]
        ia365_after_shock = df_filtered[23]

        hla = df_filtered[24]

        # LA after shock formula = LA after shock (original) - original factors + new factors
        la_updated_30 = la30_after_shock - ((df_filtered[35] * 0.85) + (df_filtered[36] * 0.95
                                                                        ) + (df_filtered[37] * 0.9) + (
                                                        df_filtered[38] * 0.85))

        la_updated_30 = la_updated_30 + ((df_filtered[35] * 0.85 * (1 + eq_shock)) + (
                df_filtered[36] * 0.95 * (1 + fin_shock)) + (df_filtered[37] * 0.9 * (1 + fild_shock))
                                         + (df_filtered[38] * 0.85 * (1 + igc_shock)))

        la_updated_90 = la90_after_shock - ((df_filtered[35] * 0.75) + (
                df_filtered[36] * 0.95) + (df_filtered[37] * 0.85) + (df_filtered[38] * 0.75))

        la_updated_90 = la_updated_90 + ((df_filtered[35] * 0.75 * (1 + eq_shock)) + (
                df_filtered[36] * 0.95 * (1 + fin_shock)) + (df_filtered[37] * 0.85 * (1 + fild_shock)) +
                                         df_filtered[38] * 0.75 * (1 + igc_shock))

        la_updated_365 = la365_after_shock - ((df_filtered[35] * 0.6) + (df_filtered[36] * 0.9
                                                                         ) + (df_filtered[37] * 0.75) + (
                                                          df_filtered[38] * 0.6))

        la_updated_365 = la_updated_365 + ((df_filtered[35] * 0.6 * (1 + eq_shock)) + (
                df_filtered[36] * 0.9 * (1 + fin_shock)) + (df_filtered[37] * 0.75 * (1 + fild_shock)) +
                                           (df_filtered[38] * 0.6 * (1 + igc_shock)))

        # IA after shock & haircut
        ia_updated_365 = ia365_after_shock - ((df_filtered[41] * 0.6 * 0.96) + (
                df_filtered[42] * 0.9 * 0.97) + (df_filtered[43] * 0.75 * 0.97) + (df_filtered[44] * 0.6 * 0.96) + (
                                                      df_filtered[40] * 0.6 * 0.96))

        ia_updated_365 = ia_updated_365 + ((df_filtered[41] * 0.6 * (1 + eq_shock) * 0.96) + (
                df_filtered[42] * 0.9 * (1 + fin_shock) * 0.97) + (df_filtered[43] * 0.75 * 0.97 * (1 + fild_shock
                                                                                                    )) \
                                           + (df_filtered[44] * (1 + igc_shock) * 0.6 * 0.96) + (
                                                   df_filtered[40] * 0.6 * (1 + ars_shock) * 0.96))

        ############################################## FIX LA & IA ##############################################

        ############################################## Phase 2 ################################################
        # slr_updated_5d = ((EQ_hedge1*0.15) +(FIN_hedge1*0.05) +(FILD_hedge1*0.1) + (currency_hedge1*0.05) + (IGC_hedge1*0.05)) - ((FIN_hedge*0.05) +(FILD_hedge*0.1) + (currency_hedge*0.05) + (IGC_hedge*0.05))
        # slr_updated_10d = (EQ_hedge1*0.15 + FIN_hedge1*0.05 +FILD_hedge1*0.1 +currency_hedge1*0.05+IGC_hedge1*0.05) - ((FIN_hedge*0.05) +(FILD_hedge*0.1) + (currency_hedge*0.05) + (IGC_hedge*0.05))
        # slr_updated_30d = (EQ_hedge1*0.15 + FIN_hedge1*0.05 +FILD_hedge1*0.1 +currency_hedge1*0.05+IGC_hedge1*0.05) - ((FIN_hedge*0.05) +(FILD_hedge*0.1) + (currency_hedge*0.05) + (IGC_hedge*0.05))
        # slr_updated_90d = (EQ_hedge1*0.25 + FIN_hedge1*0.05 +FILD_hedge1*0.1 +currency_hedge1*0.1+IGC_hedge1*0.05) - (FIN_hedge*0.05 +FILD_hedge*0.1 +currency_hedge*0.1+IGC_hedge*0.05)
        # slr_updated_365d = (EQ_hedge1*0.4 + FIN_hedge1*0.05 +FILD_hedge1*0.1 +currency_hedge1*0.2+IGC_hedge1*0.05) - (FIN_hedge*0.05 +FILD_hedge*0.1 +currency_hedge*0.2+IGC_hedge*0.05)
        #######################################################################################################

        ci_5 = df_filtered[61]
        ci_10 = df_filtered[62]
        ci_30 = df_filtered[63]
        ci_90 = df_filtered[64]
        ci_365 = df_filtered[65]

        co_5 = df_filtered[66]
        co_10 = df_filtered[67]
        co_30 = df_filtered[68]
        co_90 = df_filtered[69]
        co_365 = df_filtered[70]

        # let EQ_hedge1 be derivative notional after shock
        # let EQ_hedge be derivative notional before shock

        # equities
        alpha_MTM_c_in_eq = 0
        alpha_MTM_c_out_eq = 0

        if EQ_hedge1 > EQ_hedge:
            alpha_MTM_c_in_eq = alpha_MTM_c_in_eq + EQ_hedge1 - EQ_hedge
            alpha_MTM_c_out_eq = alpha_MTM_c_out_eq + 0

        elif EQ_hedge > EQ_hedge1:
            alpha_MTM_c_in_eq = alpha_MTM_c_in_eq + 0
            alpha_MTM_c_out_eq = alpha_MTM_c_out_eq + EQ_hedge - EQ_hedge1

        else:
            pass


        # fixed income nominal
        alpha_MTM_c_in_fin = 0
        alpha_MTM_c_out_fin = 0

        if FIN_hedge1 > FIN_hedge:
            alpha_MTM_c_in_fin = alpha_MTM_c_in_fin + FIN_hedge1 - FIN_hedge
            alpha_MTM_c_out_fin = alpha_MTM_c_out_fin + 0

        elif FIN_hedge > FIN_hedge1:
            alpha_MTM_c_in_fin = alpha_MTM_c_in_fin + 0
            alpha_MTM_c_out_fin = alpha_MTM_c_out_fin + FIN_hedge - FIN_hedge1

        else:
            pass

        # fixed income long durational
        alpha_MTM_in_FILD = 0
        alpha_MTM_out_FILD = 0

        if FILD_hedge1 > FILD_hedge:
            alpha_MTM_in_FILD = alpha_MTM_in_FILD + FILD_hedge1 - FILD_hedge
            alpha_MTM_out_FILD = alpha_MTM_out_FILD + 0

        elif FILD_hedge > FILD_hedge1:
            alpha_MTM_in_FILD = alpha_MTM_in_FILD + 0
            alpha_MTM_out_FILD = alpha_MTM_out_FILD + FILD_hedge - FILD_hedge1

        else:
            pass

        # currency
        alpha_MTM_in_FX = 0
        alpha_MTM_out_FX = 0

        if currency_hedge1 > currency_hedge:
            alpha_MTM_in_FX = alpha_MTM_in_FX + currency_hedge1 - currency_hedge
            alpha_MTM_out_FX = alpha_MTM_out_FX + 0

        elif currency_hedge > currency_hedge1:
            alpha_MTM_in_FX = alpha_MTM_in_FX + 0
            alpha_MTM_out_FX = alpha_MTM_out_FX + currency_hedge - currency_hedge1

        else:
            pass

        # investment grade credit
        alpha_MTM_in_igc = 0
        alpha_MTM_out_igc = 0

        if IGC_hedge1 > IGC_hedge:
            alpha_MTM_in_igc = alpha_MTM_in_igc + IGC_hedge1 - IGC_hedge
            alpha_MTM_out_igc = alpha_MTM_out_igc + 0

        elif IGC_hedge > IGC_hedge1:
            alpha_MTM_in_igc = alpha_MTM_in_igc + 0
            alpha_MTM_out_igc = alpha_MTM_out_igc + IGC_hedge - IGC_hedge1

        else:
            pass

        MTM_eq = alpha_MTM_c_in_eq - alpha_MTM_c_out_eq
        MTM_fin = alpha_MTM_c_in_fin - alpha_MTM_c_out_fin
        MTM_fild = alpha_MTM_in_FILD - alpha_MTM_out_FILD
        MTM_fx = alpha_MTM_in_FX - alpha_MTM_out_FX
        MTM_igc = alpha_MTM_in_igc - alpha_MTM_out_igc

        MTM_agg = MTM_eq + MTM_fin + MTM_fild + MTM_fx + MTM_igc

        if MTM_agg > 0:
            ci_5 = ci_5 + MTM_agg
        elif MTM_agg < 0:
            co_5 = co_5 + MTM_agg
        else:
            pass


        # original LCRs
        lcr5 = df_filtered[0]
        lcr10 = df_filtered[1]
        lcr30 = df_filtered[2]
        lcr90 = df_filtered[3]
        lcr365 = df_filtered[4]

        # calculated LCRs
        new_lcr_5 = (hla + ci_5) / (slr_5d - (co_5))
        new_lcr_10 = (hla + ci_5 + ci_10) / (slr_10d - (co_5 + co_10))
        new_lcr_30 = (hla + la_updated_30 + ci_5 + ci_10 + ci_30) / (slr_30d - (
                co_5 + co_10) - (co_30) * 1.2)
        new_lcr_90 = (hla + la_updated_90 + ci_5 + ci_10 + ci_30 + ci_90) / (
                slr_90d - (co_5 + co_10) - (co_30 + co_90) * 1.2)
        new_lcr_365 = (hla + la_updated_365 + ia_updated_365 + ci_5 + ci_10 +
                       ci_30 + ci_90 + ci_365) / (slr_365d - (co_5 + co_10) - (co_30 + 
                                                                 co_90 + co_365) * 1.2)

        # test, 5d
        print("##################\n")
        print("HLA: ", hla)
        print("CI5: ", ci_5)
        print("SLR5: ", slr_5d)
        print("CO5: ", co_5)
        print("LCR5: ", new_lcr_5)
        print("##################\n")

        # test, 10d
        print("CI10: ", ci_10)
        print("SLR10: ", slr_10d)
        print("CO10: ", co_10)
        print("LCR10: ", new_lcr_10)
        print("##################\n")

        # test, 30d
        print("LA 30 days: ", la_updated_30)
        print("CI30: ", ci_30)
        print("SLR30: ", slr_30d)
        print("CO30: ", co_30)
        print("LCR30: ", new_lcr_30)
        print("##################\n")

        # test, 90d
        print("LA 90 days: ", la_updated_90)
        print("CI90: ", ci_90)
        print("SLR90: ", slr_90d)
        print("CO90: ", co_90)
        print("LCR90: ", new_lcr_90)
        print("##################\n")

        # test, 365d
        print("LA 365 days: ", la_updated_365)
        print("IA 365 days: ", ia_updated_365)
        print("CI365: ", ci_365)
        print("SLR365: ", slr_365d)
        print("CO365: ", co_365)
        print("LCR365: ", new_lcr_365)
        print("##################\n\n")

        ########################################## Fix ###################################################

        return lcr5, lcr10, lcr30, lcr90, lcr365, new_lcr_5, new_lcr_10, new_lcr_30, new_lcr_90, \
            new_lcr_365, eq_initial, fin_initial, fild_initial, igc_initial, ars_initial, \
            EQ_hedge, FIN_hedge, FILD_hedge, IGC_hedge, currency_hedge, \
            alpha_MTM_c_in_eq, alpha_MTM_c_out_eq, alpha_MTM_c_in_fin, alpha_MTM_c_out_fin, \
            alpha_MTM_in_FILD, alpha_MTM_out_FILD, alpha_MTM_in_FX, alpha_MTM_out_FX, \
            alpha_MTM_in_igc, alpha_MTM_out_igc, MTM_eq, MTM_fin, MTM_fild, MTM_fx, MTM_igc, \
            equities, FIN, FILD, IGC, ARS


class CashOutflowBuffer(tk.Toplevel):
    def __init__(self, parent, expected_date_str, data_handler):
        tk.Toplevel.__init__(self, parent)
        self.expected_date_str = expected_date_str
        self.data_handler = data_handler
        self.geometry("1320x640")
        label = tk.Label(self, text="Cash Outflow Buffer Change",
                         font=("Arial", 16))
        label.grid(padx=10, pady=10)

        #input variables setter
        self.five = tk.IntVar()
        self.ten = tk.IntVar()
        self.three_zero = tk.IntVar()
        self.nine_zero = tk.IntVar()
        self.year = tk.IntVar()

        #input labels
        adj_label0 = tk.Label(self, text="Timeframes", font=("Helvetica", 16, "bold"))
        adj_label0.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W, columnspan=4)

        adj_lb1 = tk.Label(self, text="Buffer %", font=("Arial", 12, "bold"))
        adj_lb1.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W, columnspan=4)


        #input fields
        inp1 = tk.Label(self, text="0-5 Days")
        inp1.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        inp1_1 = tk.Entry(self, textvariable=self.five)
        inp1_1.grid(row=2, column=1, pady=5, padx=10, sticky=tk.W)

        inp2 = tk.Label(self, text="6-10 Days")
        inp2.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        inp2_1 = tk.Entry(self, textvariable=self.ten)
        inp2_1.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)

        inp3 = tk.Label(self, text="11-30 Days")
        inp3.grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        inp3_1 = tk.Entry(self, textvariable=self.three_zero)
        inp3_1.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)

        inp4 = tk.Label(self, text="31-90 Days")
        inp4.grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
        inp4_1 = tk.Entry(self, textvariable=self.nine_zero)
        inp4_1.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)

        inp5 = tk.Label(self, text="91-365 Days")
        inp5.grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
        inp5_1 = tk.Entry(self, textvariable=self.year)
        inp5_1.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)

        
        #additional formatting
        submit_button = tk.Button(self, text="Submit", command=self.submit_inputs)
        submit_button.grid(row=7, column=0, pady=10, columnspan=12)
        self.bind("<Return>", lambda event: submit_button.invoke())

        self.textbox1 = tk.Text(self, width=70, height=20)
        self.textbox1.grid(row=11, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)

        self.textbox2 = tk.Text(self, width=37, height=10)
        self.textbox2.grid(row=11, column=8, columnspan=13, padx=10, pady=10, sticky=tk.NSEW)

        scrollbar = tk.Scrollbar(self, command=self.textbox1.yview)
        scrollbar.grid(row=11, column=22, rowspan=3, sticky=tk.NS)
        self.textbox1.config(yscrollcommand=scrollbar.set)

        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]

        locale.setlocale(locale.LC_ALL, '')
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), 
                               round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6),
                               round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }

        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" 
            if isinstance(x, int) or isinstance(x, float) else x
        )

        self.treeview = ttk.Treeview(self, columns=list(df.columns), show='headings')
        for col in df.columns:
            self.treeview.heading(col, text=col, anchor='w')
            self.treeview.column(col, width=120, anchor='w')

        for _, row in df.iterrows():
            self.treeview.insert('', 'end', values=list(row))
            
        self.treeview.grid(row=11, column=0, columnspan=8, padx=10, pady=10, sticky=tk.NSEW)
        self.formatted_df = self.format_dataframe()


    def format_dataframe(self):
        # Create the dataframe as you did before
        df_mod = self.data_handler.dataframe
        o = df_mod.loc[:, self.expected_date_str]
        data = {
            '<HLAs>': ['Cash', 'Demand Deposit', 'Redeemable GICs', 'Short-Term Money Market',
                        '', '', '', '', ''],
            '<Exposure HLA>': [round(o[25] / 1e6), round(o[26] / 1e6), round(o[27] / 1e6), 
                               round(o[28] / 1e6),
                               '', '', '', '', ''],
            '<LAs>': ['Repos', 'Term Deposit >1D <=1M', 'Non-Redeemable GICs >1D <=1M',
                      'Term Deposits >1M <=3M',
                      "Non-Redeemable GICs >1M <=3M", "Passive Equities", "Passive FI Nominal",
                      "Passive FI Long Durational", "Passive Corporate Credits"],
            '<Exposure LA>': [round(o[30] / 1e6), round(o[31] / 1e6), round(o[32] / 1e6), 
                              round(o[33] / 1e6),
                              round(o[34] / 1e6), round(o[35] / 1e6), round(o[36] / 1e6), 
                              round(o[37] / 1e6),
                              round(o[38] / 1e6)],
            '<IAs>': ['Active ARS', 'Active Equities', 'Active FI Nominals',
                      'Active FI Long Durationals', 'Active Corporate Credits', '', '', '', ''],
            '<Exposure IA>': [round(o[40] / 1e6), round(o[41] / 1e6), round(o[42] / 1e6),
                              round(o[43] / 1e6), round(o[44] / 1e6), '', '', '', ''],
            '<Derivative Notionals>': ['Currencies', 'Equities', 'FI Nominal', 
                                       'FI Long Durational',
                                       'Investment Grade Credits', '', '', '', ''],
            '<Exposure DN>': [round(o[51] / 1e6), round(o[52] / 1e6), round(o[53] / 1e6),
                              round(o[54] / 1e6), round(o[55] / 1e6), '', '', '', '']
        }
        df = pd.DataFrame(data)
        df = df.applymap(
            lambda x: f"${locale.format_string('%.0f', x, grouping=True)}m" if isinstance(x, int) or isinstance(x,
                                                                                                                float) else x)
        return df


    def submit_inputs(self):
        lcr5, lcr10, lcr30, lcr90, lcr365, lcr5_c, lcr10_c, lcr30_c, lcr90_c, \
            lcr365_c = self.cashoutflow_buffer()

        self.textbox2.delete(1.0, tk.END)
        self.textbox2.insert(tk.END, f"Date: {self.expected_date_str}\n\n")

        self.textbox2.insert(tk.END, "*LCRs (before change | after change)*\n")
        self.textbox2.insert(tk.END,
                             f"LCR 5: {locale.format_string('%.2f', lcr5)} | {locale.format_string('%.2f', lcr5_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 10: {locale.format_string('%.2f', lcr10)} | {locale.format_string('%.2f', lcr10_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 30: {locale.format_string('%.2f', lcr30)} | {locale.format_string('%.2f', lcr30_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 90: {locale.format_string('%.2f', lcr90)} | {locale.format_string('%.2f', lcr90_c)}\n")
        self.textbox2.insert(tk.END,
                             f"LCR 365: {locale.format_string('%.2f', lcr365)} | {locale.format_string('%.2f', lcr365_c)}\n\n")


    def cashoutflow_buffer(self): 

        df = self.data_handler.dataframe
        df_filtered = df.loc[:, self.expected_date_str]

        # HLAs
        cash = df_filtered[25]
        demandDeposit = df_filtered[26]
        redeemableGIC = df_filtered[27]
        short_term_money_market = df_filtered[28]

        # LAs
        repo = df_filtered[30]
        termDeposit_1d_1m = df_filtered[31]
        non_redeemableGIC_1d_1m = df_filtered[32]
        termDeposit_1m_3m = df_filtered[33]
        non_redeemableGIC_1m_3m = df_filtered[34]
        passiveEQ = df_filtered[35]
        passiveFIN = df_filtered[36]
        passiveFILD = df_filtered[37]
        passiveIGC = df_filtered[38]

        # IAs
        activeARS = df_filtered[40]
        activeEQ = df_filtered[41]
        activeFIN = df_filtered[42]
        activeFILD = df_filtered[43]
        corporateCredit = df_filtered[44]

        # derivative notionals
        EQ_hedge = df_filtered[52]
        FIN_hedge = df_filtered[53]
        FILD_hedge = df_filtered[54]
        IGC_hedge = df_filtered[55]
        currency_hedge = df_filtered[51]

        amount_5d = df_filtered[66]
        amount_10d = df_filtered[67]
        amount_30d = df_filtered[68]
        amount_90d = df_filtered[69]
        amount_365 = df_filtered[70]

        
        modified_buffer_rate_5 = int(self.five.get()) / 100
        modified_buffer_rate_10 = int(self.ten.get()) / 100
        modified_buffer_rate_30 = int(self.three_zero.get()) / 100
        modified_buffer_rate_90 = int(self.nine_zero.get()) / 100
        modified_buffer_rate_365 = int(self.year.get()) / 100

        # day 5-10 adjustable from here
        day5 = amount_5d * (1 + modified_buffer_rate_5)
        day10 = amount_10d * (1 + modified_buffer_rate_10)
        day30 = amount_30d * (1 + modified_buffer_rate_30)
        day90 = amount_90d * (1 + modified_buffer_rate_90)
        day365 = amount_365 * (1 + modified_buffer_rate_365)

        lcr5 = df_filtered[0]
        lcr10 = df_filtered[1]
        lcr30 = df_filtered[2]
        lcr90 = df_filtered[3]
        lcr365 = df_filtered[4]


        lcr_new_5 = df_filtered[5] / (df_filtered[45] - day5)
        lcr_new_10 = df_filtered[7] / (df_filtered[46] - (day10 + day5))
        lcr_new_30 = df_filtered[9] / (df_filtered[47] - (day10 + day5 + day30))
        lcr_new_90 = df_filtered[11] / (df_filtered[48] - (day10 + day5 + day30 + day90))
        lcr_new_365 = df_filtered[13] / (df_filtered[49] - (day10 + day5 + day30 + day90 + day365))

        return lcr5, lcr10, lcr30, lcr90, lcr365, lcr_new_5, lcr_new_10, lcr_new_30, lcr_new_90, lcr_new_365

    def format_number(self, number):
        rounded_number = round(number)
        formatted_number = "{:,.0f}".format(rounded_number)
        return formatted_number

    def back_to_start_page(self):
        self.destroy()  # Close the CashOutflowBuffer window
        self.master.deiconify()


if __name__ == "__main__":
    main_menu = MainMenu()
    main_menu.mainloop()

