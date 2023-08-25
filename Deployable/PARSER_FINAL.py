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
    for ind in range(6, 15):
        LAdat = df_moto.iloc[ind]
        try:
            LAdat = LAdat.astype(float)
            LA = LA + LAdat
        except:
            False

    IA = 0
    for ind in range(16, 21):
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

    derivative_notionals = df.loc[47:51]
    derivative_notionals = derivative_notionals.reset_index(drop=True)


    frames = []
    for elmt in df:
        df_ll = df.loc[:, [elmt]]
        df_moto = classified.loc[:, [elmt]]
        df_cashin = cash_ins.loc[:, [elmt]]
        df_cashout = cash_outs.loc[:, [elmt]]
        passive_cc1 = passive_corporate_credit.loc[:, [elmt]]
        passive_cc2 = passive_cc1.iloc[0]
        passive_cc = float(passive_cc2.iloc[0])

        #cash placement
        df_notionals = derivative_notionals.loc[:, [elmt]]

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

        # calculate cash assets
        HLA, LA, IA = get_asset_in_aggregate(df_moto)
        #print("check!!!!!!!: ", LA)

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
        
        #lcr_checker(df.iloc[1][0], df.iloc[2][0], df.iloc[3][0], df.iloc[4][0],
                    #df.iloc[5][0], LCR_5, LCR_10, LCR_30, LCR_90, LCR_365)


        # LCR numerator & denominator calculation
        lcr_5_num = (HLA + CI5)
        lcr_5_denom = (SLR_5 - CO5)

        lcr_10_num = (HLA + CI5 + CI10)
        lcr_10_denom = (SLR_10 - (CO5 + CO10))

        lcr_30_num = (HLA + LA_30_after_shock + CI5 + CI10 + CI30)
        lcr_30_denom = (SLR_30 - CO5 - CO10 - CO30 * 1.2)

        lcr_90_num = (HLA + LA_90_after_shock + CI5 + CI10 + CI30 + CI90)
        lcr_90_denom = (SLR_90 - (CO5 + CO10) - (CO30 + CO90) * 1.2)

        lcr_365_num = (HLA + LA_365_after_shock + IA_365_after_shock_n_haircut + CI5 + CI10 + CI30 + CI90 + CI365)
        lcr_365_denom = (SLR_365 - CO5 - CO10 - (CO30 + CO90 + CO365) * 1.2)

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
                lcr_5_num[0], 
                lcr_5_denom[0], #
                lcr_10_num[0], 
                lcr_10_denom[0], #
                lcr_30_num[0], 
                lcr_30_denom[0], #
                lcr_90_num[0],
                lcr_90_denom[0],#
                lcr_365_num[0],
                  lcr_365_denom[0],#

                # LCR capacity
                lcr_capacity_5[0], 
                lcr_capacity_10[0], 
                lcr_capacity_30[0], 
                lcr_capacity_90[0],

                # HLA + LA + IA Before Shock & Haircuts #
                (HLA[0] + LA[0] + IA[0]),#

                # LA after shock
                LA_30_after_shock[0], LA_90_after_shock[0], LA_365_after_shock[0],

                # IA after shock & haircut
                IA_365_after_shock_n_haircut[0],

                # HLAs
                HLA[0], df_moto.iloc[0][0], df_moto.iloc[1][0], df_moto.iloc[2][0], df_moto.iloc[3][0],

                # LA before shock
                #
                LA[0], #
                df_moto.iloc[6][0], df_moto.iloc[7][0], df_moto.iloc[8][0], df_moto.iloc[9][0], df_moto.iloc[10][0],
                df_moto.iloc[11][0], df_moto.iloc[12][0], df_moto.iloc[13][0], passive_cc, 

                # IA before shock & haircut
                IA[0], #
                df_moto.iloc[16][0], df_moto.iloc[17][0], df_moto.iloc[18][0], df_moto.iloc[19][0],
                df_moto.iloc[20][0],

                # SLA aggregates
                SLR_5[0], SLR_10[0], SLR_30[0], SLR_90[0], SLR_365[0],

                # Derivative notional before shock
                notioxnal_total[0], df_notionals.iloc[0][0], df_notionals.iloc[1][0], df_notionals.iloc[2][0],
                df_notionals.iloc[3][0], df_notionals.iloc[4][0],

                # Net Flow Before Buffer
                (CI5[0] + CO5[0]), 
                (CI10[0] + CO10[0]), (CI30[0] + CO30[0]), 
                (CI90[0] + CO90[0]), 
                (CI365[0] + CO365[0]), 

                # Cash Inflow Aggregated
                CI5[0], CI10[0], CI30[0], CI90[0], CI365[0],

                # Cash Outflow Aggregated
                CO5[0], 
                CO10[0], 
                CO30[0], 
                CO90[0], 
                CO365[0],

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
        'Net Flow Before Buffer 0BD-5BD', 'Net Flow Before Buffer 6BD-10BD', 
        'Net Flow Before Buffer 11BD-30CD', 'Net Flow Before Buffer 31CD-90CD', 
        'Net Flow Before Buffer 91CD-365CD',

        # cash inflow aggregate
        'Cash Inflow 0BD-5BD', 'Cash Inflow 6BD-10BD', 'Cash Inflow 11BD-30CD', 'Cash Inflow 31CD-90CD',
        'Cash Inflow 91CD-365CD',

        # cash outflow aggregate
        'Cash Outflow Before Buffer 0BD-5BD', 'Cash Outflow Before Buffer 6BD-10BD', 
        'Cash Outflow Before Buffer 11BD-30CD', 'Cash Outflow Before Buffer 31CD-90CD',
        'Cash Outflow Before Buffer 91CD-365CD',

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