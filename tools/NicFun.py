import numpy as np 

class EquationSystemDefinition:
    def __init__(self, Bx, Px, K):
        # self.selected_qps = selected_qps.copy().reset_index(drop=True)
        self.psi_x = Px
        self.beta_x = Bx
        self.KL_true = K
        self.E = len(Px)

    def build_A_full(self):
        A = np.zeros((4, self.E))
        A[0,:] = self.beta_x * np.sin(self.psi_x) * np.cos(self.psi_x)
        A[1,:] = self.beta_x * np.cos(self.psi_x)**2
        A[2,:] = self.beta_x * np.sin(self.psi_x)**2
        A[3,:] = self.beta_x * np.sin(self.psi_x) * np.cos(self.psi_x)
        return np.sum(A * self.KL_true, axis = 1)

    def build_Q_full(self):
        Q_full = np.zeros((4, self.E, self.E))
        for i in range(self.E):
            for j in range(i):
                Q_full[0,i,j] = self.KL_true[i]*np.cos(self.psi_x[i])*self.beta_x[i]*np.sin(self.psi_x[j])*np.sin(self.psi_x[i]-self.psi_x[j])*self.beta_x[j]*self.KL_true[j]
                Q_full[1,i,j] = self.KL_true[i]*np.cos(self.psi_x[i])*self.beta_x[i]*np.cos(self.psi_x[j])*np.sin(self.psi_x[i]-self.psi_x[j])*self.beta_x[j]*self.KL_true[j]
                Q_full[2,i,j] = self.KL_true[i]*np.sin(self.psi_x[i])*self.beta_x[i]*np.sin(self.psi_x[j])*np.sin(self.psi_x[i]-self.psi_x[j])*self.beta_x[j]*self.KL_true[j]
                Q_full[3,i,j] = self.KL_true[i]*np.sin(self.psi_x[i])*self.beta_x[i]*np.cos(self.psi_x[j])*np.sin(self.psi_x[i]-self.psi_x[j])*self.beta_x[j]*self.KL_true[j]

        return np.sum(np.sum(Q_full, axis = 2), axis = 1)


# ======================================================
# 5. EJECUCIÓN DEL SISTEMA 6x6
# ======================================================

# sysdef = EquationSystemDefinition(selected_qps)
# A_full = sysdef.build_A_full()
# Q_full = sysdef.build_Q_full()
#
# constants = A_full + Q_full
