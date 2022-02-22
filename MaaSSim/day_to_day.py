from .traveller import travellerEvent
from .driver import driverEvent
import numpy as np
import pandas as pd
import math
import random


def driver_opt_out(veh, **kwargs): # user defined function to represent agent decisions
    sim = veh.sim
    working_U = veh.veh.expected_income
    #sim.logger.info("")
    #print("working_U..................", working_U)
    not_working_U = veh.veh.res_wage
    #print("not_working_U..............", not_working_U)
    working_P = (math.exp(working_U))/(math.exp(working_U)+math.exp(not_working_U))
    #print("working_p..................", working_P)
    if  working_P < random.uniform(0, 1):
        return True
    else:
        return False


def exp_income(sim):
    params = sim.params 
    sim.inData.vehicles.mu = sim.inData.vehicles.apply(lambda row: 1 if row['expected_income'] > row['res_wage'] 
                                                       else 0, axis=1) #update mu for further calculations
    run_id = sim.run_ids[-1]
    act_income = sim.res[run_id].veh_exp.PROFIT
    if sim.inData.vehicles.mu.sum() == 0:
        ave_income = 0
    else:
        ave_income = act_income.sum()/sim.inData.vehicles.mu.sum()
    # update the expected_income
    sim.inData.vehicles.expected_income = (1-params.d2d.omega)*sim.inData.vehicles.expected_income +\
                                              params.d2d.omega*sim.inData.vehicles.mu*act_income + \
                                              params.d2d.omega*(1-sim.inData.vehicles.mu)*ave_income
    sim.income.expected['run {}'.format(run_id+1)] = sim.inData.vehicles.expected_income.copy()
    sim.income.actual['run {}'.format(run_id)] = act_income.copy()


def d2d_kpi_veh(*args,**kwargs):

    """
    calculate vehicle KPIs (global and individual)
    the assumption is that driver is getting paid only for the distance between pick-up point and drop off point
    """
    
    sim =  kwargs.get('sim', None)
    params = sim.params
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event==driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['DRIVING_TIME'] = ret.ARRIVES_AT_PICKUP + ret.ARRIVES_AT_DROPOFF
    ret['DRIVING_DIST'] = ret['DRIVING_TIME']*(params.speeds.ride/1000)  #here we assume the speed is constant on the network
    ret['REVENUE'] = (ret.ARRIVES_AT_DROPOFF * (params.speeds.ride/1000) * params.platforms.fare).add(params.platforms.base_fare * ret.nRIDES) * (1-params.platforms.comm_rate)
    ret['COST'] = ret['DRIVING_DIST'] * (params.d2d.fuel_cost) # Operating Cost (OC)
    ret['PROFIT'] = ret['REVENUE'] - ret['COST']
    ret = ret[['nRIDES','nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'PROFIT', 'OUT'] + [_.name for _ in driverEvent]].fillna(0) 
    ret.index.name = 'veh'
    
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}

    
        

    
################################################################################################################
    # if len(sim.run_ids) == 0:
    #     sim.inData.vehicles.expected_income = 21 #np.random.normal(10, 1, params.nV)
    #     act_income = [0 for i in range(params.nV)]
    #     run_id = -1
    
# df = df[df['event'].isin(['IS_ACCEPTED_BY_TRAVELLER', 'ARRIVES_AT_PICKUP', 'DEPARTS_FROM_PICKUP'])]


# def driver_experience():  # kpi_veh = driver_experience
#     run_id = sim.run_ids[-1]
    
    
def my_function(veh, **kwargs): # user defined function to represent agent decisions
    sim = veh.sim
    if  veh.veh.expected_income < sim.params.d2d.res_wage:
        return True
    else:
        return False