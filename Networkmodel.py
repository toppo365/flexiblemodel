import numpy as np
import copy, random
from collections import defaultdict

class HopField:
    def __init__(self, nb_nodes):
        self.n_node = nb_nodes
        self.f = 0.5
        self.net = np.zeros((nb_nodes, nb_nodes))
        self.pattern = dict()
        self.prior = np.zeros(nb_nodes) + 0.5
        self.counter = []
        
    def add_data(self, key, arr):
        if len(arr) < self.n_node:
            self.pattern[key] = set(arr)
            data = np.zeros((1, self.n_node))
            data[0, arr] = 1
        else:
            self.pattern[key] = set(list(np.where(arr > 0)[0]))
            data = arr.reshape(1,-1)
        self.memorize(data)
        
    def memorize(self, data):
        self.net += np.dot(data.T - self.f, data - self.f)  
        for i in range(self.net.shape[0]):
            self.net[i, i] = 0 
        
    def forget(self, key):
        a, b = (int(float(key.split("-")[0])), int(key.split("-")[1]))
        tmp = [k for k in self.pattern.keys() if (a,b) == (int(float(k.split("-")[0])), int(k.split("-")[1]))]
        if len(tmp) == 0:
            print("no dict key in pattern", key)
            return 
        key = tmp[0]
        data = np.zeros((1, self.n_node))
        data[:,list(self.pattern[key])] = 1
        self.net -= np.dot(data.T - self.f, data - self.f)
        for k in tmp:
            self.pattern.pop(k)
        for i in range(self.net.shape[0]):
            self.net[i, i] = 0 

    def remember(self, init, HCprev = None,beta = 0.5, prior = None):
        if prior is None:
            prior = np.ones(self.n_node)*beta
        self.prior = prior*1
        count = 0
        while count < 1:
            energy = np.inf
            self.nodes = np.floor(np.random.rand(self.n_node) + 0.5)
            self.nodes[~np.isnan(init)] = init[~np.isnan(init)]
            pattern = []
            for k in range(50):
                self.nodes = np.floor((np.dot(self.net, self.nodes - self.f) - self.prior) > 0)    
                new_energy = self.compute_energy()
                if new_energy == energy:
                    pattern = np.where(self.nodes > 0)[0]
                    break
                else:
                    energy = new_energy
            if set(pattern) in self.pattern.values():
                tmp = [key for key,val in self.pattern.items() if val == set(pattern)]
                self.counter.append(k)
                return pattern, tmp[0]
            count += 1
        self.counter.append(50)
        return [], "0-0"

    def compute_energy(self):
        return -0.5*np.dot(self.nodes, np.dot(self.net, self.nodes - self.f)) \
                    + np.dot(self.prior, self.nodes - self.f)

class NetworkModel_org():
    def __init__(self, nHFpre, nHFpost, HC_nodes, HFC_nodes, maxcond):
        self.nstm = nHFpre
        self.HF_nodes = nHFpre + nHFpost
        self.HC_nodes = HC_nodes
        self.maxcond = maxcond
        self.nHC = 1
        self.nHF = int(self.HF_nodes/2)
        self.NGthresh = 0.7
        self.NGtagthresh = 0.7
        self.lr = 0.15
        self.HCthresh = 0.01
        self.initW = 0.3
        self.HCnet = np.zeros((HC_nodes, HC_nodes)) + self.initW - np.eye(HC_nodes)*self.initW
        self.rwd = defaultdict(list)
        self.HFnet = HopField(self.HF_nodes)
        self.HF2HC = np.zeros((HC_nodes, HFC_nodes))
        self.HC2HF = np.zeros((HFC_nodes, HC_nodes))
        if self.HF_nodes < HFC_nodes:
            print("HFC_nodes is too large")
        self.fromHF = np.sort(np.random.choice(np.arange(self.HF_nodes), HFC_nodes, replace = False))
        self.toHF = np.sort(np.random.choice(np.arange(self.HF_nodes), HFC_nodes, replace = False))
        self.NGtrans = defaultdict(float)
        self.tested = []
        self.trial = 0
        self.oddcounter = defaultdict(lambda: [0,0,0,0])
        self.escape = [] #correlated stimulus input 
        self.task_condition()
        self.condact = {k:[] for k in self.actdict.keys()} 
        
    def task_condition(self):
        pass

    def check_rwd(self, result, count):
        pass

    def memorize(self, cond, actcond):
        Sinput = self.HFregister[self.stmdict[cond][self.stmindex[cond]]]
        pattern = np.zeros((self.HF_nodes,1))
        pattern[np.random.choice(np.arange(len(Sinput), self.HF_nodes), int((self.HF_nodes - len(Sinput))/2), replace = False)] = 1
        pattern[:len(Sinput),0] = Sinput
        self.HF = np.where(pattern > 0)[0]
        key = "{}-{}".format(self.stmdict[cond][self.stmindex[cond]], actcond)
        self.HFregister[key] = pattern
        self.HFnet.add_data(key, pattern)
            
    def remember(self, cond = np.nan, HCprev = [], count = True, flag = 0):
        initstate = np.zeros(self.HF_nodes)*np.nan
        stmkind = self.stmdict[cond][self.stmindex[cond]]
        cand = sorted([k for k in self.HFnet.pattern.keys() if k.startswith(str(stmkind))])
        if count:
            self.oddcounter[self.trial][0] += 1
        infHF = np.floor(np.sum(self.HC2HF[:,HCprev[-1]], 1) > 0) if len(HCprev) > 0 else np.zeros(self.HC2HF.shape[0])  #input from H to C 
        if np.sum(infHF) == 0:  #the hippocampal input to cortex does not exist
            if cond in self.cortex_calc and (len(cand) > 0 or stmkind in self.escape): #the second or later visit of the initial state
                infHF *= np.nan
                flag = 1
            else:  #the first visit of the task state
                if count:
                    self.oddcounter[self.trial][3] += 1
                return [], "0-0", 2
        tmp = self.HFregister[stmkind]
        initstate[self.toHF] = infHF
        if np.sum(np.isnan(infHF)) > 0:  #landmark based computation
            if self.verbose:
                print("external stm", stmkind)
            initstate[:len(tmp)] = tmp
        HF, key = self.HFnet.remember(initstate)
        if len(HF) == 0:  #conversion failed
            if count:
                self.oddcounter[self.trial][2] += 1
            if len(cand) > 0:
                key = cand[0]
                return sorted(list(self.HFnet.pattern[key])), key, flag
            return [], "0-0", flag + 2
        elif set(HF[HF < len(tmp)]) != set(np.where(tmp > 0)[0]) and stmkind not in self.escape: #wrong stim. conversion
            if count:
                self.oddcounter[self.trial][1] += 1 
            return [], "0-0", flag + 4
        else:
            return HF, key, flag
        
    def forget(self, cond, maxcond):
        rmlist = [k for k in self.HFnet.pattern.keys() if k not in self.tested] 
        random.shuffle(rmlist)
        for key in rmlist:
            a, b = np.array(key.split("-"), dtype=float)
            if int(a) == cond and len(self.condact[int(a)]) > len(self.actdict[int(a)]) and b > 0:
                HC = self.pkupHC(list(self.HFnet.pattern[key]), [])
                self.HFnet.forget(key)
                self.condact[int(a)][int(b)] = -3
                preHC = np.where(self.HCnet[HC[0],:]>self.initW)[0]
                postHC = np.where(self.HCnet[:,HC[0]]>self.initW)[0]
                self.HF2HC[HC[0],:] = 0
                self.HC2HF[:,HC[0]] = 0
                self.HCnet[HC[0],preHC] = self.initW
                self.HCnet[postHC,HC[0]] = self.initW
                print("forget", key, HC, preHC)
                if np.sum(np.array(self.condact[cond]) != -3) >= maxcond:
                    return

    def infer_context(self, HCstate):
        infHF = np.floor(np.dot(self.HC2HF, HCstate) > 0)
        if np.sum(infHF) == 0:
            return np.array([0,0],dtype=int), []
        initstate = np.zeros(self.HF_nodes)*np.nan
        initstate[self.toHF] = infHF
        ptn = np.where(initstate > 0)[0]
        tmp = [(key,np.sum(np.isin(ptn, list(val)))) for key,val in self.HFnet.pattern.items()]
        if len(tmp) == 0:
            return np.array([0,0],dtype=int), []
        key,_ = max(tmp, key = lambda x: x[1])
        HF = sorted(list(self.HFnet.pattern[key]))
        return np.array(key.split("-"), dtype=float), HF

    def preplay(self, HC, key):
        inftrans = [np.array(key.split("-"), dtype=float)]
        HCstate = np.zeros(self.HCnet.shape[0])
        HCstate[HC] = 1
        prior = (np.sum(self.HC2HF > 0, 0) > 0)*(-1e-6) #to prevent from recruiting already picked up cells
        #prior = (self.HCnet[HC[0],:]>self.initW)*(-1e-6)
        HCstate_, HC_ = self.get_wta((self.HCnet - self.initW)/(1-self.initW), HCstate, self.nHC, prior = prior)
        HCtrans = [HC, HC_]
        self.learn_HC2HC(HCtrans, 1, incontext = True)
        while True:
            tmp = [self.rwd[x][-1] if x in self.rwd.keys() else 0 for x in HC_]
            exprwd = np.mean(tmp)
            HCstate, HC = self.get_wta((self.HCnet - self.initW)/(1-self.initW), HCstate_, self.nHC,
                           prior = prior, thresh = self.HCthresh, stop = True)
            if len(HC) > 0:
                inftrans_, _ = self.infer_context(HCstate)
            else:
                inftrans_, _ = self.infer_context(HCstate_)
            if inftrans_[0] != inftrans[-1][0] and inftrans_[0] != 0 and not(HC in HCtrans) and exprwd == 0:
                inftrans.append(inftrans_)
                if len(HC) == 0 or int(inftrans_[0]) == self.initial:
                    break
            else:
                break
            HCstate_, HC_ = self.get_wta((self.HCnet - self.initW)/(1-self.initW), HCstate, self.nHC, 
                            prior = prior, thresh = self.HCthresh, stop = True)
            if len(HC_) == 0 or HC_ in HCtrans:
                break
            else:
                HCtrans.extend([HC, HC_])
        return np.array(inftrans), HCtrans, exprwd
    
    def choose_action(self, key, infcond, HCtrans, HC):
        conds = infcond[:,0]
        if self.verbose:
            print("choose", conds, np.array(HCtrans).squeeze())
        for c in range(len(conds)-1):
            cond, cond2 = int(conds[c]), int(conds[c+1])
            if cond2 not in self.actdict[cond] or cond in self.terminal \
                 or (conds[c+1] != self.stmdict[cond2][self.stmindex[cond2]]): 
                infcond = infcond[:c+1,:]
                if cond in self.terminal:
                    HCtrans = HCtrans[:(c+1)*2]
                    break
                if (conds[c+1] != self.stmdict[cond2][self.stmindex[cond2]]) and len(HCtrans) >= (c+1)*2:
                    if len(HCtrans) > (c+1)*2:
                        self.learn_HC2HC(HCtrans[(c+1)*2-1:(c+1)*2+1], 0)
                        HCng = HCtrans[(c+1)*2]
                    else:
                        HCng = [-1]
                    nextcond = int(self.stmdict[cond2][self.stmindex[cond2]])
                    HC, infcond, HCtrans = self.stmremapping(HCtrans[(c+1)*2-1], HCng, nextcond, HCtrans[:(c+1)*2], infcond)
                else:
                    HCtrans = HCtrans[:(c+1)*2]
                if self.verbose:
                    print("remove", infcond[:,0], np.array(HCtrans).squeeze())
                break  
        if len(infcond) > 1:
            return HC, infcond[-1,0], HCtrans, infcond
        elif not all(infcond[0,:] == 0) and self.condact[int(infcond[0,0])][int(infcond[0,1])] != -3:
            nextcond = self.condact[int(infcond[0,0])][int(infcond[0,1])]
            return HC, nextcond, HCtrans, np.array([infcond[0,:], np.array([nextcond,-1])])
        else:
            nextcond = np.random.choice(self.actdict[int(conds[0])],1)[0]
            return HC, nextcond, HCtrans, np.array([infcond[0,:], np.array([nextcond,-1])])
            
    def pkupHC(self, act, HCprev):
        HFstate = np.zeros(self.HF_nodes)
        HFstate[act] = 1
        #prior = np.zeros(self.HC_nodes)
        prior = (np.sum(self.HC2HF > 0, 0) > 0)*(-1e-6)
        _, HC = self.get_wta(self.HF2HC, HFstate[self.fromHF], self.nHC, prior = prior)
        return HC
        
    def reward(self, cond, rwd, exprwd, HCtrans, infcond):
        if not cond     in self.terminal:
            return
        rwd_ = exprwd + self.lr*(rwd - exprwd)*int(rwd > exprwd) + self.lr*(rwd - exprwd)*(rwd < exprwd)
        for h in HCtrans[-1]:
            self.rwd[h].append(rwd_)
        if rwd > exprwd:
            self.learn_HC2HC(HCtrans, rwd)
            if rwd - exprwd < 0.3 and infcond[0,0] == self.initial:
                self.tested.extend(["{}-{}".format(a,int(b)) for a,b in infcond])
                self.tested = list(set(self.tested))
                if self.verbose:
                    print("tested",self.tested)
        if exprwd - rwd > self.HCthresh:
            self.set_NG(infcond)
        self.check_trans(HCtrans, rwd, infcond)
            
    def set_NG(self, infcond):
        self.NGtrans = defaultdict(float, {n:v*self.NGthresh for n,v in self.NGtrans.items()})
        self.NGtrans[tuple(infcond[-1,:])] += 1
        print("set NG", infcond[-1,:], {n:v for n,v in self.NGtrans.items() if v > 0})
                
    def check_trans(self, HCtrans, rwd, infcond):
        trans = []
        h = len(HCtrans)
        while h >= 0:
            h -= 1
            if np.sum(self.HC2HF[:,HCtrans[h]] > 0) == 0:
                continue
            HCstate = np.zeros(self.HC_nodes)
            HCstate[HCtrans[h]] = 1
            key, HF = self.infer_context(HCstate)
            if len(trans) > 0 and trans[0] == key[0]:
                continue
            trans[:0] = [key[0]]
            if len(HF) > 0 and h > 0 and rwd > 0:
                h -= 1
        if rwd > 0:
            self.NGtrans[tuple(infcond[-1,:])] = 0
        if self.verbose:
            print("check trans", trans, {n:v for n,v in self.NGtrans.items() if v > 0})

    def get_wta(self, net, state, num, prior = None, thresh = 0, stop = False):
        state_ = np.dot(net, state) + 1e-6
        if prior is not None:
            state_ += prior
        tmp = state_[state_ > thresh]
        if len(tmp) < num:
            if stop:
                return np.zeros_like(state_), []
            tmp2 = np.random.random(state_.size)
            pkup = np.lexsort((tmp2, state_))[-num:]
        else:
            arg = np.where(state_ > thresh)[0]
            pkup = np.random.choice(arg, num, replace = False, p = tmp/np.sum(tmp))
        state2 = np.zeros_like(state_)
        state2[pkup] = 1
        return state2, pkup

    def fancy_index(self, mat, arr1, arr2, val, method = "="):
        tmp = np.arange(mat.size).reshape(mat.shape)
        tmp2 = tmp[arr1,:][:,arr2].flatten()
        flat_mat = mat.flatten()
        if method == "=":
            flat_mat[tmp2] = val
        elif method == "+":
            flat_mat[tmp2] += val
        return flat_mat.reshape(mat.shape)
    
    def get_HFindex(self, HF):
        key_ = [k for k,v in self.HFnet.pattern.items() if v == set(HF)]
        key = key_[0] if len(key_) == 1 else "0-0"
        return key
            
    def act2index(self, act):
        return np.where(np.isin(self.fromHF, act))[0], np.where(np.isin(self.toHF, act))[0]
        
    def learn_HC2HC(self, HCtrans, rwd, incontext = False):
        H0 = self.lr*0.5
        for h in range(len(HCtrans)-1):
            w_ = self.HCnet[HCtrans[h+1], HCtrans[h]][0]
            rwd = np.max([rwd, self.initW])
            w2 = rwd if incontext else w_+self.lr*(rwd-w_)
            if w_ <= self.initW:
                self.HCnet = self.fancy_index(self.HCnet, HCtrans[h+1], HCtrans[h], w2 + H0, method = "=")
                self.HCnet[HCtrans[h+1],:] -= H0
            else:
                self.HCnet = self.fancy_index(self.HCnet, HCtrans[h+1], HCtrans[h], w2, method = "=")
        self.HCnet = np.clip(self.HCnet, 0, 1)
    
    def learn_HF2HC(self, HC, act, key = None):
        X0 = self.nHF - 0.5
        if len(act) > 0:
            index1, index2 = self.act2index(act)
            if key is not None and self.verbose:
                print("HF2HC", HC, key)
            self.HF2HC = self.fancy_index(self.HF2HC, HC, index1, X0+0.5, method = "=")
            self.HF2HC[HC,:] -= X0
            
    def learn_HC2HF(self, HC, act, key = None):
        X0 = self.nHF - 0.5
        if len(act) > 0:
            index1, index2 = self.act2index(act)
            if key is not None and self.verbose:
                print("HC2HF", HC, key)
            self.HC2HF = self.fancy_index(self.HC2HF, index2, HC, X0+0.5, method = "=")
            self.HC2HF[:,HC] -= X0
    
    def print_result(self, result):
        for i in range(len(result)):
            print(["{}:{}".format(key[0],val) for key, val in result[i].items()])
        print()
        
    def think(self, cond, HCprev, stmremap = False):
        exprwds, HCs, infstates, HCtranses, HFs, keys, NGs = [], [], [], [], [], [], []
        NGinf = []
        if self.verbose:
            print("HCprev",HCprev)
        landmark = 0
        for i in range(9):
            if i == 0 or landmark % 2 == 1:
                HF, key, landmark = self.remember(cond = cond, HCprev = HCprev, flag = landmark)  #infer context
                HC = self.pkupHC(HF, HCprev)
                if self.verbose:
                    print("think", key, HC)
                if landmark > 1:
                    break
            if len(HF) != self.nHF:
                continue
            infcond_, HCtrans_, exprwd_ = self.preplay(HC, key)
            NG = 0
            if self.NGtrans[tuple(infcond_[-1,:])] > self.NGtagthresh:
                NG = -self.NGtrans[tuple(infcond_[-1,:])]
            elif exprwd_ > 0.05:
                self.HF = HF
                return HC, exprwd_, infcond_, HCtrans_, key
            HCs.append(HC)
            exprwds.append(exprwd_)
            infstates.append(infcond_)
            HCtranses.append(HCtrans_)
            HFs.append(HF)
            keys.append(key)
            NGs.append(NG)
        if len(exprwds) > 0:
            exprwd_ = np.random.choice(len(exprwds),1)[0]
            infcond = infstates[exprwd_]
            HCtrans = HCtranses[exprwd_]
            #bifurcation candidates
            clen_ = [c for c in range(int(len(HCtrans)/2)) \
                       if len(self.actdict[int(infcond[c,0])])>np.sum(self.HCnet[:,HCtrans[c*2]]>self.initW+self.HCthresh)]
            #not to choose the worst choice
            if len(clen_) == 0 and np.min(NGs) == NGs[exprwd_] < 0 and np.max(NGs) != NGs[exprwd_]:
                exprwd_ = np.where(np.array(NGs)!=NGs[exprwd_])[0][0]
            infcond = infstates[exprwd_]
            HCtrans = HCtranses[exprwd_]
            #bifurcation candidates
            clen_ = [c for c in range(int(len(HCtrans)/2)) \
                       if len(self.actdict[int(infcond[c,0])])>np.sum(self.HCnet[:,HCtrans[c*2]]>self.initW+self.HCthresh)]
            #multiple options
            option = [c for c in range(int(len(HCtrans)/2)) if len(self.actdict[int(infcond[c,0])]) > 1]
            exprwd = exprwds[exprwd_]    
            HC = HCs[exprwd_]  
            self.HF = HFs[exprwd_]
            key = keys[exprwd_]
            NG = NGs[exprwd_]
            flag1 = NG < 0 and len(option) > 0   #multiple action options and NG on
            flag2 = len(clen_) > 0 and len(infcond) > 1  #contextual state < the no. of action
            if flag2 and (flag1 or np.random.rand() < 0.3):
                clen = clen_[-1]
                NGinf = []
                if clen+1 < len(infcond):
                    NGinf.append(int(infcond[clen+1,0]))
                infcond = infcond[:clen+1,:]
                HCtrans = HCtrans[:(clen+1)*2]
                nextcond = self.plan_action(int(infcond[-1,0]), NGinf)
                infcond = np.vstack((infcond, np.array([[nextcond, np.nan]])))           
                return self.actremapping(infcond, HC, HCtrans, key) #action driven remapping
            return HC, np.abs(exprwd), infcond, HCtrans, key
        else:        #no converged context
            if not stmremap:
                nextcond = self.plan_action(cond, [])
                return self.make_newcontext(cond, nextcond)   #initial context preparation
            else:
                return [],[],[],[],[]

    def actremapping(self, infcond, HC, HCtrans, key):
        exprwd = 0
        pivot = HCtrans[-2]
        HCstate = np.zeros(self.HCnet.shape[0])
        HCstate[pivot] = 1
        prior = np.zeros(self.HC_nodes)
        prior[self.HCnet[:,pivot].squeeze() > self.HCthresh + self.initW] = -self.HF_nodes
        prior += (np.sum(self.HC2HF > 0, 0) > 0)*(-1e-6)

        _, HC_ = self.get_wta((self.HCnet - self.initW)/(1-self.initW), HCstate, self.nHC, prior = prior)

        HCtrans2 = HCtrans[:-1] + [HC_]
        self.condact[int(infcond[-2,0])][int(infcond[-2,1])] = infcond[-1,0]
        self.learn_HC2HC(HCtrans2[-2:], 1, incontext = True)
        print("actremapping", np.array(HCtrans).squeeze(), np.array(HCtrans2).squeeze())
        return HC, exprwd, infcond, HCtrans2, key
                    
    def plan_action(self, cond, NGinf):
        nextcand = np.array(self.actdict[cond])[~np.isin(self.actdict[cond], NGinf)]
        if len(nextcand) == 1:
            nextcond = nextcand[0]
        else:
            tmp = np.isin(self.actdict[cond], self.condact[cond])
            if all(tmp):
                tmpcount = [self.condact[cond].count(i) for i in self.actdict[cond]]
                tmp = np.random.random(len(tmpcount))
                for i in np.lexsort((tmp, tmpcount)):
                    nextcond = self.actdict[cond][i]
                    if len(self.actdict[cond]) == 1 or not nextcond in NGinf:
                        break
            else:
                nextcond = self.actdict[cond][np.random.choice(np.where(~tmp)[0],1)[0]]
        if self.verbose:
            print("plan", cond, NGinf, nextcond)
        return nextcond

    def make_newcontext(self,cond,cond2):
        if -2 in self.condact[cond]:
            print("already exist", cond, cond2)
            return 
        if cond2 not in self.actdict[cond]:
            print("context not found", cond, cond2)
            return
        maxcond = self.maxcond if self.maxcond is not None else len(self.actdict[cond])+1
        if np.sum(np.array(self.condact[cond]) != -3) >= maxcond:
            self.forget(cond, maxcond)
            if np.sum(np.array(self.condact[cond]) != -3) >= maxcond:
                print("cannot make more context", cond, cond2)
                return
        self.condact[cond].append(-2)
        self.initial_context(cond, cond2)
        key = self.get_HFindex(self.HF)
        HC = self.pkupHC(self.HF, [])
        self.learn_HF2HC(HC, self.HF, key)
        self.learn_HC2HF(HC, self.HF, key)
        if self.verbose:
            print("pkup", HC, key)
        infcond, HCtrans, exprwd = self.preplay(HC, key)
        if len(infcond) == 1:
            infcond = np.vstack((infcond, [cond2, np.nan]))
        return HC, exprwd, infcond, HCtrans, key
    
    def initial_context(self, cond, nextcond):
        if nextcond not in self.actdict[cond]:
            print("next cond not in actdict", cond, nextcond)
            return False
        elif np.sum(np.isin(self.condact[cond],[-2,nextcond])) == 0:            
            self.condact[cond].append(-2)
        actcond = np.where(np.array(self.condact[cond]) == -2)[0]
        if len(actcond) == 0:
            return False
        print("init", cond, nextcond)
        actcond = actcond[0]
        self.memorize(cond, actcond)
        self.condact[cond][actcond] = nextcond
        if self.verbose:
            print("memo", cond, actcond)
        return True

    def stmremapping(self, HC_, HCng, cond, HCtrans_, infcond_):
        arg = np.where(self.HCnet[:,HC_]>self.HCthresh+self.initW)[0]
        if self.verbose:
            print("stmremap",HCng, HC_, arg)
        arg = arg[arg != HCng]
        arg = np.append(np.random.permutation(arg),HC_)
        for x in arg:
            HC, exprwd, infcond, HCtrans, key = self.think(cond = cond, HCprev = [[x]], stmremap = x != arg[-1])
            if len(HC) == 0:
                continue
            target = 0.65 if self.HCnet[HC,HC_] < 0.65 else self.HCnet[HC,HC_][0]
            self.learn_HC2HC([HC_, HC], target)
            self.learn_HC2HF([x], self.HF, str(infcond[0,:]))
            infcond = np.vstack((infcond_, infcond))
            HCtrans = HCtrans_+ HCtrans
            return HC, infcond, HCtrans

    def process(self, trial, HCprev_, cond = None, verbose = False):
        self.trial = trial
        self.verbose = verbose
        result = []
        if cond == self.initial:
            HCprev = [HCprev_[-1]] if len(HCprev_) > 0 else []
        elif cond is None:
            cond, HCprev = self.initial, []
        else:
            HCprev = copy.deepcopy(HCprev_)

        infcond = np.array([np.array([-1,-1])])
        HCtrans = []
        nextcond = -1
        if self.verbose:
            print("trial", trial, "stmindex", self.stmindex[2], "HCprev", np.array(HCprev).squeeze())
        while len(result) < 15:
            self.HF = []
            if cond in self.terminal and len(HCprev) >= len(infcond[:,1])*2 and infcond[-1,0] == nextcond:
                self.HF, key, _ = self.remember(cond = cond, HCprev = [HCprev[-3]], count = False)
                HC = HCprev[-2]
                nextcond = self.actdict[cond][0] 
                flag = False
            else:
                HC, exprwd, infcond_, HCtrans, key = self.think(cond, HCprev)
                HC, nextcond, HCtrans, infcond = self.choose_action(key, infcond_, HCtrans, HC) 
                infcond = infcond[~np.isnan(infcond[:,1]),:]
                flag = True
            tmp = HCprev.index(HCtrans[0]) if HCtrans[0] in HCprev else len(HCprev)
            HCtrans2 = HCprev[:tmp] + HCtrans
            result.append({"cond":cond, "nextcond":nextcond, "exprwd":exprwd, "infcond":list(infcond[:,0]), "context":list(infcond[:,1]),\
                           "preinf":list(infcond_[:,0]),"HCtrans":np.array(HCtrans2).squeeze(),"act":key})
            if flag:
                HFkey = self.get_HFindex(self.HF)
                if len(HCprev) > 0 and HCtrans[0] == HC:
                    self.learn_HC2HF(HCprev[-1], self.HF, HFkey)
                self.learn_HF2HC(HC, self.HF, HFkey)
                self.learn_HC2HF(HC, self.HF, HFkey)
            rwd = self.check_rwd(result, trial)
            result[-1]["rwd"] = rwd
            if ~np.isnan(rwd):
                self.reward(cond, rwd, exprwd, HCtrans2, infcond)
            if self.initial is None:
                if self.verbose:
                    print("initial is none", self.actdict[cond][0], nextcond, cond)
                self.initial = self.actdict[cond][0] 
                cond = None
            else:
                cond = int(nextcond)
            if infcond[-1,0] == nextcond:
                if nextcond in self.terminal and len(HCtrans2) > len(infcond[:,0])*2:
                    HCprev = HCtrans2
                else: 
                    HCprev = HCtrans2[-len(HCtrans):]
            else:
                HCprev = [HCtrans2[-1]]
            if len(result) == 15:
                cond = None
                HCprev = []
            if result[-1]["cond"] in self.terminal:
                break
        result[-1]["pattern"] = list(self.HFnet.pattern.keys())

        if verbose:
            self.print_result(result)

        return result, HCprev, cond