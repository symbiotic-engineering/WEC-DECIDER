import openmdao.api as om

from geometryComponent import geometryComponent
from dynamicsComponent import dynamicsComponent
from structureComponent import structureComponent
from econComponent import econComponent
class waveEngergy(om.Group):

    def setup(self):
        self.add_subsystem('geometryComponent', geometryComponent())
        self.add_subsystem('dynamicsComponent', dynamicsComponent())
        self.add_subsystem('structureComponent', structureComponent())
        self.add_subsystem('econComponent', econComponent())

    def configure(self):
        pass
        #self.promotes('comp2', inputs=['a'], outputs=['b'])

top = om.Problem(model=waveEngergy())
top.setup()

print(top.list_indep_vars())



#class waveEnergy(om.Group):
#    pass


"""
    om.IndepVarComp
    om.ExecComp ("b * 2")
    om.ExplicitComponent
"""
"""
    # input
    in_params = p.copy()
    in_params['D_f'] = X[0]
    D_s_over_D_f = X[1]
    h_f_over_D_f = X[2]
    T_s_over_h_s = X[3]
    in_params['F_max'] = X[4] * 1e6
    in_params['B_p'] = X[5] * 1e6
    in_params['w_n'] = X[6]
    #Change float to Int
    in_params['M'] = int(X[7])
    
    #output
    LCOE , P_Var, G (output)
    """
