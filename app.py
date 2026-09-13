import math
import io
import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="MLRweb — ML Workbench", page_icon="ML", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#f7f9fc}.topbar{background:#123f63;color:white;padding:14px 20px;border-radius:6px;margin-bottom:12px}.brand{font-size:24px;font-weight:800}.developer{font-size:12px;margin-top:4px;opacity:.95}.section-title{color:#173f60;font-weight:800;border-left:4px solid #173f60;padding-left:8px;margin-top:8px;margin-bottom:10px}.ready{display:inline-block;background:#e7f6ec;color:#15753a;border:1px solid #b9e2c5;padding:5px 12px;border-radius:18px;font-weight:700}
</style>
<div class="topbar"><div class="brand">MLRweb &nbsp; ML Workbench — Regression Platform</div><div class="developer">Developed by <b>Dr. Furquan Ahmad</b> · GitHub Edition</div></div>
""", unsafe_allow_html=True)

MODEL_NAMES = {
    "LR":"Linear Regression","Ridge":"Ridge Regression","Lasso":"Lasso Regression","EN":"Elastic Net","DTR":"Decision Tree","ABR":"AdaBoost","GBR":"Gradient Boosting","HGBR":"Hist GradBoost","RFR":"Random Forest","ETR":"Extra Trees","BGR":"Bagging","SVR":"Support Vector (SVR)","KNN":"K-Nearest Neighbors","XGB":"XGBoost","MLP":"Neural Network (MLP)","ELM":"Extreme Learning Machine (ELM)",
    "ANFIS":"ANFIS (FCM Neuro-Fuzzy)","PSO-ANN":"PSO-ANN","GA-ANN":"GA-ANN","GWO-ANN":"GWO-ANN","MPA-ANN":"MPA-ANN (Marine Predators)","OOA-ANN":"OOA-ANN (Osprey)","ANFIS-MPA":"ANFIS-MPA","ANFIS-OOA":"ANFIS-OOA","ANFIS-ELM":"ANFIS-ELM", "CGF-ANN":"CGF-ANN (Fletcher–Reeves Conjugate Gradient)"
}

for key, default in [("results",None),("trained_models",{}),("predictions",{}),("convergence",{})]:
    if key not in st.session_state: st.session_state[key] = default


def get_excel_sheets(f): return pd.ExcelFile(f).sheet_names

def read_excel_sheet(f, sheet): return pd.read_excel(f, sheet_name=sheet)


def safe_max_depth(v): return None if v is None or int(v) <= 0 else int(v)


def make_models(p, seed):
    models={
        "LR":LinearRegression(), "Ridge":Ridge(alpha=p["Ridge"]),
        "Lasso":Lasso(alpha=p["Lasso"][0],max_iter=p["Lasso"][1],random_state=seed),
        "EN":ElasticNet(alpha=p["EN"][0],l1_ratio=p["EN"][1],max_iter=p["EN"][2],random_state=seed),
        "DTR":DecisionTreeRegressor(max_depth=safe_max_depth(p["DTR"][0]),min_samples_leaf=p["DTR"][1],min_samples_split=p["DTR"][2],random_state=seed),
        "ABR":AdaBoostRegressor(n_estimators=p["ABR"][0],learning_rate=p["ABR"][1],loss=p["ABR"][2],random_state=seed),
        "GBR":GradientBoostingRegressor(n_estimators=p["GBR"][0],learning_rate=p["GBR"][1],max_depth=p["GBR"][2],subsample=p["GBR"][3],random_state=seed),
        "HGBR":HistGradientBoostingRegressor(learning_rate=p["HGBR"][0],max_depth=safe_max_depth(p["HGBR"][1]),max_iter=p["HGBR"][2],min_samples_leaf=p["HGBR"][3],random_state=seed),
        "RFR":RandomForestRegressor(n_estimators=p["RFR"][0],max_depth=safe_max_depth(p["RFR"][1]),min_samples_leaf=p["RFR"][2],random_state=seed,n_jobs=-1),
        "ETR":ExtraTreesRegressor(n_estimators=p["ETR"][0],max_depth=safe_max_depth(p["ETR"][1]),min_samples_leaf=p["ETR"][2],random_state=seed,n_jobs=-1),
        "BGR":BaggingRegressor(n_estimators=p["BGR"][0],max_samples=p["BGR"][1],max_features=p["BGR"][2],random_state=seed,n_jobs=-1),
        "SVR":SVR(kernel=p["SVR"][0],C=p["SVR"][1],epsilon=p["SVR"][2],degree=p["SVR"][3]),
        "KNN":KNeighborsRegressor(n_neighbors=p["KNN"][0],weights=p["KNN"][1],p=p["KNN"][2]),
        "ELM":ELMRegressor(hidden=p["ELM"][0],activation=p["ELM"][1],alpha=p["ELM"][2],seed=seed),
        "MLP":MLPRegressor(hidden_layer_sizes=(p["MLP"][0],p["MLP"][1]),activation=p["MLP"][2],alpha=p["MLP"][3],max_iter=p["MLP"][4],random_state=seed)
    }
    if XGB_AVAILABLE:
        models["XGB"]=xgb.XGBRegressor(n_estimators=p["XGB"][0],max_depth=p["XGB"][1],learning_rate=p["XGB"][2],alpha=p["XGB"][3],subsample=p["XGB"][4],colsample_bytree=p["XGB"][5],random_state=seed,verbosity=0,objective="reg:squarederror",n_jobs=-1)
    return models


class ANFISRegressor:
    """Compact first-order Takagi-Sugeno ANFIS with FCM initialization and hybrid learning."""
    def __init__(self,n_clusters=5,m=2.0,fcm_iterations=200,threshold=1e-4,epochs=200,error_goal=0.0,step_size=.01,step_decrease=1.1,seed=1):
        self.n_clusters=int(n_clusters); self.m=float(m); self.fcm_iterations=int(fcm_iterations); self.threshold=float(threshold); self.epochs=int(epochs); self.error_goal=float(error_goal); self.step_size=float(step_size); self.step_decrease=float(step_decrease); self.seed=int(seed)
    def _fcm(self,X):
        rng=np.random.default_rng(self.seed); n=len(X); k=self.n_clusters
        U=rng.random((n,k)); U/=U.sum(axis=1,keepdims=True)
        prev=None
        for _ in range(self.fcm_iterations):
            Um=U**self.m; centers=(Um.T@X)/(Um.sum(axis=0)[:,None]+1e-12)
            d=np.sqrt(((X[:,None,:]-centers[None,:,:])**2).sum(axis=2))+1e-12
            U_new=1/(d[:,:,None]/d[:,None,:])**(2/(self.m-1))
            if prev is not None and np.max(np.abs(U_new-U))<self.threshold: U=U_new; break
            U=U_new
            prev=centers.copy()
        return centers,U
    def _forward(self,X):
        z=-0.5*np.sum(((X[:,None,:]-self.centers[None,:,:])/self.sigmas[None,:,:])**2,axis=2)
        z=np.clip(z,-60,20); w=np.exp(z); a=w/(w.sum(axis=1,keepdims=True)+1e-12)
        A=np.c_[X,np.ones(len(X))]; f=A@self.theta.T
        return a,(a*f).sum(axis=1),f,z
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float); self.n_features=X.shape[1]
        self.centers,U=self._fcm(X)
        spread=np.sqrt(((X[:,None,:]-self.centers[None,:,:])**2*U[:,:,None]).sum(axis=0)/(U.sum(axis=0)[:,None]+1e-12))
        self.sigmas=np.maximum(spread, np.std(X,axis=0,keepdims=True)*0.05+1e-3)
        rng=np.random.default_rng(self.seed); self.theta=rng.normal(0,.01,(self.n_clusters,self.n_features+1))
        lr=self.step_size; self.history=[]
        A=np.c_[X,np.ones(len(X))]
        for epoch in range(self.epochs):
            alpha,pred,f,z=self._forward(X)
            # Consequent least-squares update (hybrid learning)
            Phi=np.einsum('nr,ni->nri',alpha,A).reshape(len(X),-1)
            try: coef=np.linalg.lstsq(Phi,y,rcond=1e-8)[0]; self.theta=coef.reshape(self.n_clusters,self.n_features+1)
            except np.linalg.LinAlgError: pass
            alpha,pred,f,z=self._forward(X); err=pred-y; mse=float(np.mean(err**2)); self.history.append(mse)
            if mse<=self.error_goal: break
            # Premise gradients: d yhat / d z_r = alpha_r(f_r-yhat)
            dz=alpha*(f-pred[:,None]); common=(2*err[:,None]/len(X))*dz
            grad_c=np.sum(common[:,:,None]*(X[:,None,:]-self.centers[None,:,:])/(self.sigmas[None,:,:]**2),axis=0)
            grad_s=np.sum(common[:,:,None]*((X[:,None,:]-self.centers[None,:,:])**2)/(self.sigmas[None,:,:]**3),axis=0)
            self.centers-=lr*grad_c; self.sigmas=np.maximum(.01, self.sigmas-lr*grad_s)
            lr=max(1e-7,lr/self.step_decrease)
        self.n_epochs_trained=len(self.history); return self
    def predict(self,X): return self._forward(np.asarray(X,float))[1]



class ELMRegressor:
    """Extreme Learning Machine with random hidden layer and closed-form output weights."""
    def __init__(self, hidden=50, activation="tanh", seed=1, alpha=1e-6):
        self.hidden=int(hidden); self.activation=activation; self.seed=int(seed); self.alpha=float(alpha)
    def _act(self,z):
        if self.activation=="relu": return np.maximum(0,z)
        if self.activation=="sigmoid": return 1/(1+np.exp(-np.clip(z,-50,50)))
        return np.tanh(z)
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float); rng=np.random.default_rng(self.seed)
        self.W=rng.normal(0,1,(X.shape[1],self.hidden)); self.b=rng.normal(0,1,self.hidden)
        H=self._act(X@self.W+self.b); I=np.eye(self.hidden)
        self.beta=np.linalg.solve(H.T@H+self.alpha*I,H.T@y)
        return self
    def predict(self,X): return self._act(np.asarray(X,float)@self.W+self.b)@self.beta

def _levy_step(shape,rng,beta=1.5):
    sigma=(math.gamma(1+beta)*np.sin(np.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=rng.normal(0,sigma,shape); v=rng.normal(0,1,shape)
    return u/(np.abs(v)**(1/beta)+1e-12)

def mpa_optimize(objective, dim, pop_size, iterations, lower, upper, seed=1):
    rng=np.random.default_rng(seed); lo=float(lower); hi=float(upper); P=0.5
    prey=rng.uniform(lo,hi,(pop_size,dim)); fit=np.array([objective(x) for x in prey]); best=prey[np.argmin(fit)].copy(); best_fit=float(fit.min()); history=[]
    old_prey=prey.copy(); old_fit=fit.copy()
    for t in range(1,iterations+1):
        elite=np.tile(best,(pop_size,1)); CF=(1-t/iterations)**(2*t/iterations)
        RB=rng.normal(size=(pop_size,dim)); RL=0.05*_levy_step((pop_size,dim),rng)
        new=prey.copy()
        for i in range(pop_size):
            if t < iterations/3:
                step=RB[i]*(elite[i]-RB[i]*prey[i]); new[i]=prey[i]+P*rng.random()*step
            elif t < 2*iterations/3:
                if i < pop_size/2:
                    step=RL[i]*(elite[i]-RL[i]*prey[i]); new[i]=prey[i]+P*rng.random()*step
                else:
                    step=RB[i]*(RB[i]*elite[i]-prey[i]); new[i]=elite[i]+P*CF*step
            else:
                step=RL[i]*(RL[i]*elite[i]-prey[i]); new[i]=elite[i]+P*CF*step
        new=np.clip(new,lo,hi); new_fit=np.array([objective(x) for x in new])
        improved=new_fit<old_fit; prey=np.where(improved[:,None],new,old_prey); fit=np.where(improved,new_fit,old_fit)
        old_prey,old_fit=prey.copy(),fit.copy()
        if fit.min()<best_fit: best=prey[np.argmin(fit)].copy(); best_fit=float(fit.min())
        # Eddy formation / FAD effect
        if rng.random()<0.2:
            U=rng.random((pop_size,dim))<0.2; prey=np.clip(prey+CF*(lo+rng.random((pop_size,dim))*(hi-lo))*U,lo,hi)
        else:
            r=rng.random(); prey=np.clip(prey+((0.2*(1-r)+r))*(prey[rng.permutation(pop_size)]-prey[rng.permutation(pop_size)]),lo,hi)
        history.append(float(np.sqrt(best_fit)))
    return best,pd.DataFrame({"Iteration":np.arange(1,iterations+1),"RMSE":history})

def ooa_optimize(objective, dim, pop_size, iterations, lower, upper, seed=1):
    rng=np.random.default_rng(seed); lo=float(lower); hi=float(upper)
    X=rng.uniform(lo,hi,(pop_size,dim)); fit=np.array([objective(x) for x in X]); best=X[np.argmin(fit)].copy(); best_fit=float(fit.min()); history=[]
    for t in range(1,iterations+1):
        new=X.copy()
        for i in range(pop_size):
            prey_idx=int(rng.integers(pop_size)); prey=X[prey_idx]
            # OOA exploration: osprey detects and dives toward a randomly selected prey.
            r=rng.random(dim); direction=prey-X[i]
            candidate=X[i]+r*direction*(1+0.5*rng.random())
            candidate=np.clip(candidate,lo,hi); f1=objective(candidate)
            # OOA exploitation: carry the prey to a safe/suitable random location;
            # perturbation decreases with iteration.
            safe=lo+rng.random(dim)*(hi-lo)
            candidate2=X[i]+(safe-X[i])/t
            candidate2=np.clip(candidate2,lo,hi); f2=objective(candidate2)
            if f1 < fit[i] and f1 <= f2: new[i]=candidate
            elif f2 < fit[i]: new[i]=candidate2
        X=new; fit=np.array([objective(x) for x in X]); j=np.argmin(fit)
        if fit[j]<best_fit: best=X[j].copy(); best_fit=float(fit[j])
        history.append(float(np.sqrt(best_fit)))
    return best,pd.DataFrame({"Iteration":np.arange(1,iterations+1),"RMSE":history})

def optimized_ann_fit(algorithm,X,y,p):
    n_in=X.shape[1]; h=p["hidden"]; dim=n_in*h+2*h+1
    objective=lambda w: ann_objective(w,X,y,n_in,h)
    if algorithm=="MPA": return mpa_optimize(objective,dim,p["population"],p["iterations"],p["lower"],p["upper"],p["seed"])
    return ooa_optimize(objective,dim,p["population"],p["iterations"],p["lower"],p["upper"],p["seed"])

def optimized_anfis_fit(algorithm,X,y,p):
    base=ANFISRegressor(n_clusters=p["n_clusters"],m=p["m"],fcm_iterations=p["fcm_iterations"],threshold=p["threshold"],epochs=p["epochs"],error_goal=p["error_goal"],step_size=p["step_size"],step_decrease=p["step_decrease"],seed=p["seed"]).fit(X,y)
    k=base.n_clusters; nf=X.shape[1]; dim=k*(2*nf+nf+1)
    def unpack(v):
        a=k*nf; b=a+k*nf; centers=v[:a].reshape(k,nf); sig=np.maximum(.01,v[a:b].reshape(k,nf)); theta=v[b:].reshape(k,nf+1); return centers,sig,theta
    def obj(v):
        centers,sig,theta=unpack(v); z=-.5*np.sum(((X[:,None,:]-centers[None,:,:])/sig[None,:,:])**2,axis=2); w=np.exp(np.clip(z,-60,20)); alpha=w/(w.sum(axis=1,keepdims=True)+1e-12); pred=(alpha*(np.c_[X,np.ones(len(X))]@theta.T)).sum(axis=1); return float(np.mean((pred-y)**2))
    vec=np.r_[base.centers.ravel(),base.sigmas.ravel(),base.theta.ravel()]; span=max(float(np.std(vec)),0.1); lo=vec-span; hi=vec+span
    if algorithm=="MPA": best,h=mpa_optimize(obj,dim,p["population"],p["iterations"],lo.mean(),hi.mean(),p["seed"])
    else: best,h=ooa_optimize(obj,dim,p["population"],p["iterations"],lo.mean(),hi.mean(),p["seed"])
    centers,sig,theta=unpack(best); base.centers=centers; base.sigmas=sig; base.theta=theta; return base,h

def ann_unpack(weights,n_in,n_hidden):
    a=n_in*n_hidden; b=a+n_hidden; c=b+n_hidden; d=c+1
    W1=weights[:a].reshape(n_in,n_hidden); b1=weights[a:b]; W2=weights[b:c]; b2=weights[c:d][0]; return W1,b1,W2,b2

def ann_predict(weights,X,n_in,n_hidden):
    W1,b1,W2,b2=ann_unpack(weights,n_in,n_hidden); H=np.tanh(X@W1+b1); return H@W2+b2


def ann_objective(weights,X,y,n_in,n_hidden):
    p=ann_predict(weights,X,n_in,n_hidden); return float(np.mean((p-y)**2))


def cgf_ann_fit(X,y,p):
    """ANN optimized by Fletcher–Reeves nonlinear conjugate gradient (CGF)."""
    rng=np.random.default_rng(p["seed"]); n_in=X.shape[1]; h=int(p["hidden"]); lo=float(p["lower"]); hi=float(p["upper"]); it=int(p["iterations"])
    w=rng.uniform(lo,hi,n_in*h+2*h+1)
    def loss_grad(v):
        W1,b1,W2,b2=ann_unpack(v,n_in,h); H=np.tanh(X@W1+b1); pred=H@W2+b2; err=pred-y
        loss=float(np.mean(err**2)); d=(2.0/len(X))*err
        gW2=H.T@d; gb2=float(np.sum(d)); dZ=(d[:,None]*W2[None,:])*(1-H**2); gW1=X.T@dZ; gb1=np.sum(dZ,axis=0)
        return loss,np.r_[gW1.ravel(),gb1,gW2,gb2],pred
    f,g,pred=loss_grad(w); d=-g; history=[_history_row(0,y,pred)]
    for t in range(1,it+1):
        if not np.all(np.isfinite(g)) or np.linalg.norm(g)<1e-10: break
        step=float(p.get("step",0.1)); gd=float(np.dot(g,d))
        if gd>=0: d=-g; gd=float(np.dot(g,d))
        accepted=False
        for _ in range(20):
            cand=np.clip(w+step*d,lo,hi); nf,ng,npred=loss_grad(cand)
            if nf <= f + 1e-4*step*gd: accepted=True; break
            step*=0.5
        if not accepted: cand=np.clip(w+step*d,lo,hi); nf,ng,npred=loss_grad(cand)
        beta=float(np.dot(ng,ng)/(np.dot(g,g)+1e-20)); beta=min(beta,10.0)
        w=cand; f=nf; g_old=g; g=ng; d=-g+beta*d; pred=npred; history.append(_history_row(t,y,pred))
        if f<=float(p.get("error_goal",0.0)): break
    return w,pd.DataFrame(history)


def cgf_ann_nh_sweep(X_train, y_train, X_test, y_test, base_cfg, nh_values=range(2,31)):
    """Run Fletcher–Reeves CGF-ANN for Nh=2..30 and return TR/TS metrics."""
    rows=[]
    histories={}
    predictions=[]
    for nh in nh_values:
        cfg=dict(base_cfg); cfg["hidden"]=int(nh)
        model,hist=cgf_ann_fit(X_train,y_train,cfg)
        ptr=ann_predict(model,X_train,X_train.shape[1],int(nh))
        pte=ann_predict(model,X_test,X_train.shape[1],int(nh))
        rows.append({"Nh":int(nh),"Model":"CGF-ANN","R2_Train":float(r2_score(y_train,ptr)),"RMSE_Train":float(np.sqrt(mean_squared_error(y_train,ptr))),"MAE_Train":float(mean_absolute_error(y_train,ptr)),"R2_Test":float(r2_score(y_test,pte)),"RMSE_Test":float(np.sqrt(mean_squared_error(y_test,pte))),"MAE_Test":float(mean_absolute_error(y_test,pte))})
        for phase, actuals, preds in (("TR", y_train, ptr), ("TS", y_test, pte)):
            for idx, (actual, pred) in enumerate(zip(actuals, preds), start=1):
                predictions.append({"Nh":int(nh),"Model":"CGF-ANN","Dataset":phase,"Sample":idx,"Actual":float(actual),"Predicted":float(pred),"Residual":float(actual-pred)})
        histories[int(nh)]=hist
    return pd.DataFrame(rows), histories, pd.DataFrame(predictions)



def mpa_ooa_population_sweep(X_train, y_train, X_test, y_test, base_cfg, architectures=((8,17,1),(8,26,1)), populations=range(10,101,10), iterations_list=(500,1000), algorithms=("MPA","OOA")):
    """Research sweep matching the manuscript-style population/iteration table.
    Runs ANN-MPA and ANN-OOA for architectures 8-17-1 and 8-26-1,
    population sizes 10..100, at 500 and 1000 iterations.
    """
    if X_train.shape[1] != 8:
        raise ValueError(f"This analysis requires exactly 8 input parameters; the dataset has {X_train.shape[1]} predictors.")
    rows=[]; histories={}
    for arch in architectures:
        n_in, nh, n_out = arch
        if n_out != 1:
            raise ValueError("Only single-output ANN architectures are supported.")
        for algorithm in algorithms:
            for iterations in iterations_list:
                for population in populations:
                    cfg=dict(base_cfg); cfg.update(hidden=int(nh), population=int(population), iterations=int(iterations))
                    # Deterministic but distinct seed for every experiment.
                    cfg["seed"]=int(base_cfg.get("seed",1))+nh*100000+population*100+iterations+(0 if algorithm=="MPA" else 5000000)
                    weights,hist = optimized_ann_fit(algorithm,X_train,y_train,cfg)
                    ptr=ann_predict(weights,X_train,n_in,nh); pte=ann_predict(weights,X_test,n_in,nh)
                    key=f"{algorithm}_{n_in}-{nh}-{n_out}_{iterations}_{population}"
                    rows.append({
                        "Architecture":f"{n_in}-{nh}-{n_out}","Model":f"ANN-{algorithm}",
                        "Population":int(population),"Iterations":int(iterations),
                        "R2_Train":float(r2_score(y_train,ptr)),"RMSE_Train":float(np.sqrt(mean_squared_error(y_train,ptr))),"MAE_Train":float(mean_absolute_error(y_train,ptr)),
                        "R2_Test":float(r2_score(y_test,pte)),"RMSE_Test":float(np.sqrt(mean_squared_error(y_test,pte))),"MAE_Test":float(mean_absolute_error(y_test,pte))
                    })
                    histories[key]=hist
    return pd.DataFrame(rows), histories

def initialize_ann_population(size,n_in,n_hidden,rng,low,high):
    dim=n_in*n_hidden+2*n_hidden+1
    return rng.uniform(low,high,(size,dim))


def _history_row(iteration, y, pred):
    return {"Iteration": int(iteration), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": float(mean_absolute_error(y, pred))}


def pso_ann_fit(X,y,p):
    rng=np.random.default_rng(p["seed"]); n_in=X.shape[1]; h=p["hidden"]; n=p["swarm"]; it=p["iterations"]; lo=p["lower"]; hi=p["upper"]; c1=p["c1"]; c2=p["c2"]; w=p["inertia"]
    pos=initialize_ann_population(n,n_in,h,rng,lo,hi); vel=np.zeros_like(pos); pbest=pos.copy(); scores=np.array([ann_objective(v,X,y,n_in,h) for v in pos]); g=pos[np.argmin(scores)].copy(); gs=float(scores.min()); history=[]
    history.append(_history_row(0,y,ann_predict(g,X,n_in,h)))
    for t in range(1,it+1):
        r1=rng.random(pos.shape); r2=rng.random(pos.shape); vel=w*vel+c1*r1*(pbest-pos)+c2*r2*(g-pos); pos=np.clip(pos+vel,lo,hi)
        s=np.array([ann_objective(v,X,y,n_in,h) for v in pos]); improved=s<scores; pbest[improved]=pos[improved]; scores[improved]=s[improved]
        j=np.argmin(scores)
        if scores[j]<gs: gs=float(scores[j]); g=pbest[j].copy()
        history.append(_history_row(t,y,ann_predict(g,X,n_in,h)))
    return g, pd.DataFrame(history)


def ga_ann_fit(X,y,p):
    rng=np.random.default_rng(p["seed"]); n_in=X.shape[1]; h=p["hidden"]; popn=p["population"]; gens=p["generations"]; lo=p["lower"]; hi=p["upper"]; cross=p["crossover"]; mut=p["mutation"]; elite=max(1,int(p["elite"]))
    pop=initialize_ann_population(popn,n_in,h,rng,lo,hi); history=[]
    for gen in range(1,gens+1):
        scores=np.array([ann_objective(v,X,y,n_in,h) for v in pop]); order=np.argsort(scores); best=pop[order[0]].copy(); history.append(_history_row(gen,y,ann_predict(best,X,n_in,h)))
        new=[pop[i].copy() for i in order[:elite]]
        inv=1/(scores-scores.min()+1e-9); prob=inv/inv.sum()
        while len(new)<popn:
            i,j=rng.choice(popn,2,p=prob); a,b=pop[i].copy(),pop[j].copy()
            if rng.random()<cross:
                mask=rng.random(a.size)<.5; child=np.where(mask,a,b)
            else: child=a
            mask=rng.random(child.size)<mut; child[mask]+=rng.normal(0,(hi-lo)*.05,mask.sum()); new.append(np.clip(child,lo,hi))
        pop=np.asarray(new)
    scores=np.array([ann_objective(v,X,y,n_in,h) for v in pop]); best=pop[np.argmin(scores)]
    return best, pd.DataFrame(history)


def gwo_ann_fit(X,y,p):
    rng=np.random.default_rng(p["seed"]); n_in=X.shape[1]; h=p["hidden"]; n=p["wolves"]; it=p["iterations"]; lo=p["lower"]; hi=p["upper"]; dim=n_in*h+2*h+1
    wolves=initialize_ann_population(n,n_in,h,rng,lo,hi); history=[]
    for t in range(1,it+1):
        scores=np.array([ann_objective(v,X,y,n_in,h) for v in wolves]); idx=np.argsort(scores); alpha,beta,delta=wolves[idx[:3]].copy(); a=2-2*(t-1)/max(1,it-1)
        history.append(_history_row(t,y,ann_predict(alpha,X,n_in,h)))
        new=[]
        for x in wolves:
            vals=[]
            for leader in (alpha,beta,delta):
                A=2*a*rng.random(dim)-a; C=2*rng.random(dim); vals.append(leader-A*np.abs(C*leader-x))
            new.append(np.clip(np.mean(vals,axis=0),lo,hi))
        wolves=np.asarray(new)
    scores=np.array([ann_objective(v,X,y,n_in,h) for v in wolves]); best=wolves[np.argmin(scores)]; history.append(_history_row(it+1,y,ann_predict(best,X,n_in,h)))
    return best, pd.DataFrame(history)


def train_hybrid(code,X,y,p):
    if code=="ANFIS":
        model=ANFISRegressor(**p).fit(X,y)
        hist=pd.DataFrame([{"Iteration":i+1,"RMSE":float(np.sqrt(m))} for i,m in enumerate(model.history)])
        return model, hist
    if code=="PSO-ANN": return (*pso_ann_fit(X,y,p),)
    if code=="GA-ANN": return (*ga_ann_fit(X,y,p),)
    if code=="GWO-ANN": return (*gwo_ann_fit(X,y,p),)
    if code=="MPA-ANN": return (*optimized_ann_fit("MPA",X,y,p),)
    if code=="CGF-ANN": return (*cgf_ann_fit(X,y,p),)
    if code=="OOA-ANN": return (*optimized_ann_fit("OOA",X,y,p),)
    if code=="ANFIS-MPA": return (*optimized_anfis_fit("MPA",X,y,p),)
    if code=="ANFIS-OOA": return (*optimized_anfis_fit("OOA",X,y,p),)
    if code=="ANFIS-ELM":
        anf=ANFISRegressor(**p).fit(X,y); residual=y-anf.predict(X); elm=ELMRegressor(hidden=p.get("elm_hidden",50),activation=p.get("elm_activation","tanh"),seed=p["seed"],alpha=p.get("elm_alpha",1e-6)).fit(X,residual); return (anf,elm,pd.DataFrame({"Iteration":[1],"RMSE":[float(np.sqrt(mean_squared_error(y,anf.predict(X)+elm.predict(X))))],"MAE":[float(mean_absolute_error(y,anf.predict(X)+elm.predict(X)))]}))


def hybrid_predict(code,model,X,n_in,h):
    if code=="ANFIS": return model.predict(X)
    if code=="ANFIS-ELM": return model[0].predict(X)+model[1].predict(X)
    return ann_predict(model,X,n_in,h)



def _base_candidate_models(seed):
    """Compact, reproducible candidate sets used for automatic tuning of the 16 base regressors."""
    grids={
        "LR":[{}],
        "Ridge":[{"alpha":a} for a in [0.01,0.1,1,10,100]],
        "Lasso":[{"alpha":a,"max_iter":10000} for a in [1e-5,1e-4,1e-3,1e-2,1e-1]],
        "EN":[{"alpha":a,"l1_ratio":r,"max_iter":10000} for a in [1e-4,1e-3,1e-2,1e-1] for r in [0.2,0.5,0.8]],
        "DTR":[{"max_depth":d,"min_samples_leaf":l,"min_samples_split":sp} for d in [None,3,5,8,12] for l in [1,2,5] for sp in [2,5,10]],
        "ABR":[{"n_estimators":n,"learning_rate":lr,"loss":loss} for n in [50,100,200] for lr in [0.03,0.1,0.3] for loss in ["linear","square","exponential"]],
        "GBR":[{"n_estimators":n,"learning_rate":lr,"max_depth":d,"subsample":ss} for n in [100,300,500] for lr in [0.03,0.1] for d in [2,3,5] for ss in [0.8,1.0]],
        "HGBR":[{"learning_rate":lr,"max_depth":d,"max_iter":it,"min_samples_leaf":leaf} for lr in [0.03,0.1] for d in [None,3,6] for it in [100,300] for leaf in [10,20]],
        "RFR":[{"n_estimators":n,"max_depth":d,"min_samples_leaf":l} for n in [100,300,500] for d in [None,5,10,20] for l in [1,5,10]],
        "ETR":[{"n_estimators":n,"max_depth":d,"min_samples_leaf":l} for n in [100,300,500] for d in [None,5,10,20] for l in [1,5,10]],
        "BGR":[{"n_estimators":n,"max_samples":ms,"max_features":mf} for n in [50,100,200] for ms in [0.7,1.0] for mf in [0.7,1.0]],
        "SVR":[{"kernel":k,"C":C,"epsilon":e,"degree":d} for k in ["rbf","linear","poly"] for C in [1,100,10000] for e in [0.01,0.1,1] for d in [2,3]],
        "KNN":[{"n_neighbors":k,"weights":w,"p":p} for k in [2,3,5,7,10,15] for w in ["uniform","distance"] for p in [1,2]],
        "XGB":[{"n_estimators":n,"max_depth":d,"learning_rate":lr,"alpha":a,"subsample":ss,"colsample_bytree":cs} for n in [100,300,500] for d in [2,4,6] for lr in [0.03,0.1] for a in [0,0.2] for ss in [0.8,1.0] for cs in [0.8,1.0]],
        "MLP":[{"hidden_layer_sizes":h,"activation":act,"alpha":a,"max_iter":3000} for h in [(50,), (100,), (100,50), (150,75)] for act in ["relu","tanh"] for a in [1e-5,1e-3,1e-2]],
        "ELM":[{"hidden":h,"activation":act,"alpha":a} for h in [20,50,100,200,400] for act in ["tanh","relu","sigmoid"] for a in [1e-8,1e-6,1e-4]],
    }
    return grids


def _make_base_from_spec(code,spec,seed):
    if code=="LR": return LinearRegression()
    if code=="Ridge": return Ridge(**spec)
    if code=="Lasso": return Lasso(random_state=seed,**spec)
    if code=="EN": return ElasticNet(random_state=seed,**spec)
    if code=="DTR": return DecisionTreeRegressor(random_state=seed,**spec)
    if code=="ABR": return AdaBoostRegressor(random_state=seed,**spec)
    if code=="GBR": return GradientBoostingRegressor(random_state=seed,**spec)
    if code=="HGBR": return HistGradientBoostingRegressor(random_state=seed,**spec)
    if code=="RFR": return RandomForestRegressor(random_state=seed,n_jobs=-1,**spec)
    if code=="ETR": return ExtraTreesRegressor(random_state=seed,n_jobs=-1,**spec)
    if code=="BGR": return BaggingRegressor(random_state=seed,n_jobs=-1,**spec)
    if code=="SVR": return SVR(**spec)
    if code=="KNN": return KNeighborsRegressor(**spec)
    if code=="XGB" and XGB_AVAILABLE: return xgb.XGBRegressor(random_state=seed,verbosity=0,objective="reg:squarederror",n_jobs=-1,**spec)
    if code=="MLP": return MLPRegressor(random_state=seed,**spec)
    if code=="ELM": return ELMRegressor(seed=seed,**spec)
    return None


def tune_16_base_models(X,y,seed,progress_cb=None):
    """Tune the 15 original models + ELM using a validation split of the training data."""
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=seed,shuffle=True)
    grids=_base_candidate_models(seed)
    codes=["LR","Ridge","Lasso","EN","DTR","ABR","GBR","HGBR","RFR","ETR","BGR","SVR","KNN"]
    if XGB_AVAILABLE: codes.append("XGB")
    codes += ["MLP","ELM"]
    best_specs={}; rows=[]
    total=sum(len(grids[c]) for c in codes); counter=0
    for code in codes:
        best_r2=-np.inf; best_spec=grids[code][0]; best_rmse=np.inf
        for spec in grids[code]:
            counter+=1
            try:
                m=_make_base_from_spec(code,spec,seed); m.fit(Xt,yt); pv=m.predict(Xv); r=r2_score(yv,pv); rm=np.sqrt(mean_squared_error(yv,pv))
                if (r>best_r2) or (np.isclose(r,best_r2) and rm<best_rmse): best_r2=float(r); best_rmse=float(rm); best_spec=dict(spec)
            except Exception:
                pass
            if progress_cb: progress_cb(counter/max(1,total))
        best_specs[code]=best_spec
        rows.append({"Model":code,"Best_Hyperparameters":str(best_spec),"Validation_R2":best_r2,"Validation_RMSE":best_rmse})
    return best_specs,pd.DataFrame(rows).sort_values("Validation_R2",ascending=False).reset_index(drop=True)


def metric_row(name,ytr,ptr,yte,pte,elapsed,extra=None):
    d={"Model":name,"R2_Train":r2_score(ytr,ptr),"RMSE_Train":np.sqrt(mean_squared_error(ytr,ptr)),"MAE_Train":mean_absolute_error(ytr,ptr),"R2_Test":r2_score(yte,pte),"RMSE_Test":np.sqrt(mean_squared_error(yte,pte)),"MAE_Test":mean_absolute_error(yte,pte),"Time_s":elapsed}
    if extra: d.update(extra)
    return d


tab1,tab2,tab3=st.tabs(["1  WORKBENCH","2  RESULTS & ANALYSIS","3  CODE GENERATOR"])

with st.sidebar:
    st.markdown("### SPLIT MODE")
    split_mode=st.radio("Split Mode",["Auto Split","Manual Split"])
    if split_mode=="Auto Split":
        test_percent=st.number_input("Test %",5,50,20,5); seed=st.number_input("Seed",0,999999,1); shuffle=st.selectbox("Shuffle",["Yes","No"])
    else:
        train_file=st.file_uploader("Training Excel",type=["xlsx"],key="train_excel"); train_sheet=None
        if train_file:
            ts=get_excel_sheets(train_file); train_sheet=st.selectbox("Training Sheet Name",ts,index=ts.index("T") if "T" in ts else 0)
        test_file=st.file_uploader("Testing Excel",type=["xlsx"],key="test_excel"); test_sheet=None
        if test_file:
            ts=get_excel_sheets(test_file); test_sheet=st.selectbox("Testing Sheet Name",ts,index=ts.index("T") if "T" in ts else 0)
        seed=st.number_input("Seed",0,999999,1,key="manual_seed"); shuffle=st.selectbox("Shuffle",["Yes","No"],key="manual_shuffle")
    st.markdown("### UPLOAD DATASET")
    if split_mode=="Auto Split":
        uploaded_file=st.file_uploader("Excel File (.xlsx)",type=["xlsx"]); sheet_name=None
        if uploaded_file:
            sheets=get_excel_sheets(uploaded_file); sheet_name=st.selectbox("Excel Sheet Name",sheets,index=sheets.index("T") if "T" in sheets else 0)
    st.markdown("### PREPROCESSING")
    scaler_name=st.selectbox("Scaler",["StandardScaler","MinMaxScaler","RobustScaler","None"]); st.caption("Recommended for SVR, KNN and MLP and ANN hybrids")

with tab1:
    left,right=st.columns([.36,.64])
    with left:
        st.markdown('<div class="section-title">MODELS & HYPERPARAMETERS</div>',unsafe_allow_html=True)
        select_all=st.checkbox("✓ ALL",value=True); selected_models=[]
        def switch(code):
            if st.checkbox(f"{code}  ·  {MODEL_NAMES[code]}",value=select_all,key=f"enable_{code}"): selected_models.append(code)
        switch("LR")
        ridge_alpha=st.number_input("Ridge alpha",.0001,100000.,1.0); switch("Ridge")
        lasso_alpha=st.number_input("Lasso alpha",.000001,100.,.001,format="%.6f"); lasso_iter=st.number_input("Lasso max_iter",100,100000,1000,100); switch("Lasso")
        en_alpha=st.number_input("EN alpha",.000001,100.,.001,format="%.6f"); en_l1=st.number_input("EN l1_ratio",0.,1.,.5); en_iter=st.number_input("EN max_iter",100,100000,1000,100); switch("EN")
        tree_depth=st.number_input("DTR max_depth (0=None)",0,100,4); tree_leaf=st.number_input("DTR min_leaf",1,100,2); tree_split=st.number_input("DTR min_split",2,100,2); switch("DTR")
        ada_n=st.number_input("ABR n_est",5,2000,45); ada_lr=st.number_input("ABR lr",.0001,2.,.35); ada_loss=st.selectbox("ABR loss",["linear","square","exponential"]); switch("ABR")
        gb_n=st.number_input("GBR n_est",5,2000,45); gb_lr=st.number_input("GBR lr",.0001,2.,.35); gb_depth=st.number_input("GBR max_depth",1,100,4); gb_subsample=st.number_input("GBR subsample",.1,1.,1.); switch("GBR")
        hgb_lr=st.number_input("HGBR lr",.0001,2.,.1); hgb_depth=st.number_input("HGBR max_depth (0=None)",0,100,4); hgb_iter=st.number_input("HGBR max_iter",10,2000,100); hgb_leaf=st.number_input("HGBR min_leaf",1,500,20); switch("HGBR")
        rf_n=st.number_input("RFR n_est",5,2000,45); rf_depth=st.number_input("RFR max_depth (0=None)",0,100,4); rf_leaf=st.number_input("RFR min_leaf",1,100,1); switch("RFR")
        et_n=st.number_input("ETR n_est",5,2000,45); et_depth=st.number_input("ETR max_depth (0=None)",0,100,4); et_leaf=st.number_input("ETR min_leaf",1,100,1); switch("ETR")
        bag_n=st.number_input("BGR n_est",5,2000,45); bag_samples=st.number_input("BGR max_samp",.1,1.,1.); bag_features=st.number_input("BGR max_feat",.1,1.,1.); switch("BGR")
        svr_kernel=st.selectbox("SVR kernel",["rbf","linear","poly","sigmoid"]); svr_c=st.number_input("SVR C",.0001,10000000.,10000.); svr_epsilon=st.number_input("SVR epsilon",0.,1000.,10.); svr_degree=st.number_input("SVR degree",1,10,4); switch("SVR")
        knn_k=st.number_input("KNN k",1,100,5); knn_weights=st.selectbox("KNN weights",["uniform","distance"]); knn_p=st.number_input("KNN p",1,5,2); switch("KNN")
        if XGB_AVAILABLE:
            xgb_n=st.number_input("XGB n_est",5,2000,45); xgb_depth=st.number_input("XGB max_depth",1,100,4); xgb_lr=st.number_input("XGB lr",.0001,2.,.35); xgb_alpha=st.number_input("XGB alpha",0.,1000.,.2); xgb_subsample=st.number_input("XGB subsample",.1,1.,1.); xgb_colsample=st.number_input("XGB col_bt",.1,1.,1.); switch("XGB")
        mlp_l1=st.number_input("MLP layer1",1,1000,100); mlp_l2=st.number_input("MLP layer2",1,1000,50); mlp_activation=st.selectbox("MLP activation",["relu","tanh","logistic"]); mlp_alpha=st.number_input("MLP alpha",.0000001,10.,.0001,format="%.7f"); mlp_iter=st.number_input("MLP max_iter",100,20000,5000,100); switch("MLP")
        elm_hidden=st.number_input("ELM hidden neurons",5,1000,100,key="base_elm_hidden"); elm_alpha=st.number_input("ELM regularization",1e-10,100.,1e-6,format="%.8f",key="base_elm_alpha"); elm_activation=st.selectbox("ELM activation",["tanh","relu","sigmoid"],key="base_elm_activation"); switch("ELM")
        auto_tune_16=st.checkbox("Auto-tune all 16 base models and report best hyperparameters",value=True)
        st.markdown("### RESEARCH / ANN–ANFIS MODELS")
        st.caption("All parameters below are editable. The 9–10–1 configuration is only the default; input neurons are detected from Excel.")
        anfis_clusters=st.number_input("ANFIS FCM clusters",2,30,5); anfis_m=st.number_input("ANFIS partition matrix m",1.1,5.,2.0,.1); anfis_fcm_iter=st.number_input("ANFIS FCM iterations",10,5000,200,10); anfis_thresh=st.number_input("ANFIS improvement threshold",1e-8,1.,1e-4,format="%.8f"); anfis_epochs=st.number_input("ANFIS epochs",1,5000,200); anfis_goal=st.number_input("ANFIS error goal",0.,1e6,0.); anfis_step=st.number_input("ANFIS initial step size",1e-6,1.,.01,format="%.6f"); anfis_decay=st.number_input("ANFIS step decrease rate",1.,10.,1.1,.05); switch("ANFIS")
        ann_h=st.number_input("ANN hidden neurons (Nh)",1,500,10); ann_low=st.number_input("ANN weight lower bound",-10.,0.,-1.); ann_high=st.number_input("ANN weight upper bound",0.,10.,1.); ann_input=st.number_input("ANN input neurons override (0=auto)",0,500,0); ann_effective_in=int(ann_input) if ann_input>0 else None
        pso_swarm=st.number_input("PSO swarm population",2,500,15); pso_iter=st.number_input("PSO iterations",1,5000,500); pso_c1=st.number_input("PSO c1",0.,5.,2.); pso_c2=st.number_input("PSO c2",0.,5.,2.); pso_inertia=st.number_input("PSO inertia",0.,2.,.7); switch("PSO-ANN")
        ga_pop=st.number_input("GA population",4,500,100); ga_gen=st.number_input("GA generations",1,5000,500); ga_cross=st.number_input("GA crossover probability",0.,1.,.90); ga_mut=st.number_input("GA mutation probability",0.,1.,.01); ga_elite=st.number_input("GA elite count",1,50,2); switch("GA-ANN")
        gwo_wolves=st.number_input("GWO wolf population",3,500,20); gwo_iter=st.number_input("GWO iterations",1,5000,500); switch("GWO-ANN")
        mpa_pop=st.number_input("MPA population",3,500,20); mpa_iter=st.number_input("MPA iterations",1,5000,500); switch("MPA-ANN")
        ooa_pop=st.number_input("OOA osprey population",3,500,20); ooa_iter=st.number_input("OOA iterations",1,5000,500); switch("OOA-ANN")
        cgf_iter=st.number_input("CGF iterations",1,5000,500); cgf_step=st.number_input("CGF initial step",1e-5,2.0,0.1,format="%.5f"); switch("CGF-ANN")
        cgf_sweep=st.checkbox("CGF-ANN hidden-neuron sweep (Nh=2–30)",value=False,key="cgf_nh_sweep")
        switch("ANFIS-MPA"); switch("ANFIS-OOA")
        anfis_elm_hidden=st.number_input("ANFIS-ELM residual hidden neurons",5,1000,50); anfis_elm_alpha=st.number_input("ANFIS-ELM residual regularization",1e-10,100.,1e-6,format="%.8f"); anfis_elm_activation=st.selectbox("ANFIS-ELM residual activation",["tanh","relu","sigmoid"]); switch("ANFIS-ELM")
    with right:
        st.markdown('<div class="section-title">UPLOAD / CONFIGURATION</div>',unsafe_allow_html=True)
        if split_mode=="Auto Split":
            if uploaded_file is None: st.info("Upload your Excel workbook from the left. The available sheet names will appear automatically.")
            else:
                preview=read_excel_sheet(uploaded_file,sheet_name); st.write(f"**Excel Sheet:** `{sheet_name}`"); c1,c2,c3=st.columns(3); c1.metric("Samples",len(preview)); c2.metric("Columns",len(preview.columns)); c3.metric("Target",str(preview.columns[-1])); st.dataframe(preview.head(10),use_container_width=True,height=250); st.caption("The last column is automatically treated as the target.")
        else: st.info("Manual Split: upload separate training and testing Excel files.")
        c1,c2,c3,c4=st.columns(4); c1.metric("Models Enabled",len(selected_models)); c2.metric("Split",split_mode); c3.metric("Scaler",scaler_name); c4.markdown('<div class="ready">● READY</div>',unsafe_allow_html=True)
        run_training=st.button("▶ RUN TRAINING",type="primary",use_container_width=True)
        if run_training:
            if split_mode=="Auto Split":
                if uploaded_file is None: st.error("Please upload an Excel (.xlsx) file."); st.stop()
                df=read_excel_sheet(uploaded_file,sheet_name); X=df.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; y=pd.to_numeric(df.iloc[:,-1],errors="coerce").values; valid=np.isfinite(X).all(axis=1)&np.isfinite(y); X=X[valid]; y=y[valid]
                if len(X)<5: st.error("At least 5 valid rows are required."); st.stop()
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_percent/100,random_state=int(seed),shuffle=shuffle=="Yes")
            else:
                if train_file is None or test_file is None: st.error("Upload both training and testing Excel files."); st.stop()
                tr=read_excel_sheet(train_file,train_sheet); te=read_excel_sheet(test_file,test_sheet); X_train=tr.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; y_train=pd.to_numeric(tr.iloc[:,-1],errors="coerce").values; X_test=te.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; y_test=pd.to_numeric(te.iloc[:,-1],errors="coerce").values
                if not (np.isfinite(X_train).all() and np.isfinite(y_train).all() and np.isfinite(X_test).all() and np.isfinite(y_test).all()): st.error("Training/testing sheets contain non-numeric or missing values in the modelling columns."); st.stop()
            if X_train.shape[1]!=X_test.shape[1]: st.error("Training and testing files must have the same number of predictor columns."); st.stop()
            scaler={"StandardScaler":StandardScaler,"MinMaxScaler":MinMaxScaler,"RobustScaler":RobustScaler,"None":None}[scaler_name]
            sc=scaler() if scaler else None
            if sc: X_train=sc.fit_transform(X_train); X_test=sc.transform(X_test)
            n_in=X_train.shape[1]
            effective_n=int(ann_input) if ann_input>0 else n_in
            if effective_n!=n_in and any(m in selected_models for m in ["PSO-ANN","GA-ANN","GWO-ANN","MPA-ANN","OOA-ANN"]): st.error(f"ANN input neurons override is {effective_n}, but the dataset has {n_in} predictors. Set override to 0 (auto) or exactly {n_in}."); st.stop()
            if any(m in selected_models for m in ["ANFIS","ANFIS-MPA","ANFIS-OOA","ANFIS-ELM"]) and anfis_clusters>len(X_train): st.error("ANFIS clusters cannot exceed the number of training samples."); st.stop()
            params={"Ridge":ridge_alpha,"Lasso":(lasso_alpha,lasso_iter),"EN":(en_alpha,en_l1,en_iter),"DTR":(tree_depth,tree_leaf,tree_split),"ABR":(ada_n,ada_lr,ada_loss),"GBR":(gb_n,gb_lr,gb_depth,gb_subsample),"HGBR":(hgb_lr,hgb_depth,hgb_iter,hgb_leaf),"RFR":(rf_n,rf_depth,rf_leaf),"ETR":(et_n,et_depth,et_leaf),"BGR":(bag_n,bag_samples,bag_features),"SVR":(svr_kernel,svr_c,svr_epsilon,svr_degree),"KNN":(knn_k,knn_weights,knn_p),"MLP":(mlp_l1,mlp_l2,mlp_activation,mlp_alpha,mlp_iter),"ELM":(elm_hidden,elm_activation,elm_alpha)}
            if XGB_AVAILABLE: params["XGB"]=(xgb_n,xgb_depth,xgb_lr,xgb_alpha,xgb_subsample,xgb_colsample)
            all_models=make_models(params,int(seed)); results=[]; fitted={}; prediction_data={}; convergence_data={}; base_actual_pred=[]; best_hyperparams=None; progress=st.progress(0); status=st.empty()
            if auto_tune_16:
                st.info("Auto-tuning the 16 base regressors on a validation split of the training data. Final reported TR/TS metrics use the selected hyperparameters retrained on the full training set.")
                tune_bar=st.progress(0)
                best_specs,best_hyperparams=tune_16_base_models(X_train,y_train,int(seed),progress_cb=lambda v:tune_bar.progress(min(1.0,max(0.0,float(v)))))
                for code,spec in best_specs.items():
                    if code in all_models: all_models[code]=_make_base_from_spec(code,spec,int(seed))
                st.session_state.best_hyperparams=best_hyperparams
            hybrid_cfg={"ANFIS":dict(n_clusters=int(anfis_clusters),m=float(anfis_m),fcm_iterations=int(anfis_fcm_iter),threshold=float(anfis_thresh),epochs=int(anfis_epochs),error_goal=float(anfis_goal),step_size=float(anfis_step),step_decrease=float(anfis_decay),seed=int(seed)),"PSO-ANN":dict(hidden=int(ann_h),swarm=int(pso_swarm),iterations=int(pso_iter),c1=float(pso_c1),c2=float(pso_c2),inertia=float(pso_inertia),lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"GA-ANN":dict(hidden=int(ann_h),population=int(ga_pop),generations=int(ga_gen),crossover=float(ga_cross),mutation=float(ga_mut),elite=int(ga_elite),lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"GWO-ANN":dict(hidden=int(ann_h),wolves=int(gwo_wolves),iterations=int(gwo_iter),lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"MPA-ANN":dict(hidden=int(ann_h),population=int(mpa_pop),iterations=int(mpa_iter),lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"OOA-ANN":dict(hidden=int(ann_h),population=int(ooa_pop),iterations=int(ooa_iter),lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"CGF-ANN":dict(hidden=int(ann_h),iterations=int(cgf_iter),step=float(cgf_step),error_goal=0.0,lower=float(ann_low),upper=float(ann_high),seed=int(seed)),"ANFIS-MPA":dict(n_clusters=int(anfis_clusters),m=float(anfis_m),fcm_iterations=int(anfis_fcm_iter),threshold=float(anfis_thresh),epochs=int(anfis_epochs),error_goal=float(anfis_goal),step_size=float(anfis_step),step_decrease=float(anfis_decay),population=int(mpa_pop),iterations=int(mpa_iter),seed=int(seed)),"ANFIS-OOA":dict(n_clusters=int(anfis_clusters),m=float(anfis_m),fcm_iterations=int(anfis_fcm_iter),threshold=float(anfis_thresh),epochs=int(anfis_epochs),error_goal=float(anfis_goal),step_size=float(anfis_step),step_decrease=float(anfis_decay),population=int(ooa_pop),iterations=int(ooa_iter),seed=int(seed)),"ANFIS-ELM":dict(n_clusters=int(anfis_clusters),m=float(anfis_m),fcm_iterations=int(anfis_fcm_iter),threshold=float(anfis_thresh),epochs=int(anfis_epochs),error_goal=float(anfis_goal),step_size=float(anfis_step),step_decrease=float(anfis_decay),elm_hidden=int(elm_hidden),elm_alpha=float(elm_alpha),elm_activation=elm_activation,seed=int(seed))}
            for i,name in enumerate(selected_models,1):
                status.info(f"Training {name} — {MODEL_NAMES[name]}"); start=time.time()
                try:
                    if name in ["ANFIS","PSO-ANN","GA-ANN","GWO-ANN","MPA-ANN","OOA-ANN","CGF-ANN","ANFIS-MPA","ANFIS-OOA","ANFIS-ELM"]:
                        model,history=train_hybrid(name,X_train,y_train,hybrid_cfg[name]); ptr=hybrid_predict(name,model,X_train,n_in,int(ann_h)); pte=hybrid_predict(name,model,X_test,n_in,int(ann_h)); convergence_data[name]=history; cfg=hybrid_cfg[name]
                        extra={"Configuration":("FCM clusters="+str(cfg["n_clusters"]) if name.startswith("ANFIS") else "Nh="+str(cfg["hidden"]))}
                    else:
                        model=all_models[name]; model.fit(X_train,y_train); ptr=model.predict(X_train); pte=model.predict(X_test); extra={}
                    elapsed=time.time()-start; results.append(metric_row(name,y_train,ptr,y_test,pte,elapsed,extra)); fitted[name]=model; prediction_data[name]=pd.DataFrame({"Observed":y_test,"Predicted":pte,"Residual":y_test-pte})
                    if name in ["LR","Ridge","Lasso","EN","DTR","ABR","GBR","HGBR","RFR","ETR","BGR","SVR","KNN","XGB","MLP","ELM"]:
                        base_actual_pred.extend([{"Model":name,"Dataset":"TR","Sample":j+1,"Actual":float(a),"Predicted":float(b),"Residual":float(a-b)} for j,(a,b) in enumerate(zip(y_train,ptr))])
                        base_actual_pred.extend([{"Model":name,"Dataset":"TS","Sample":j+1,"Actual":float(a),"Predicted":float(b),"Residual":float(a-b)} for j,(a,b) in enumerate(zip(y_test,pte))])
                except Exception as e:
                    results.append({"Model":name,"R2_Train":np.nan,"RMSE_Train":np.nan,"MAE_Train":np.nan,"R2_Test":np.nan,"RMSE_Test":np.nan,"MAE_Test":np.nan,"Time_s":time.time()-start,"Error":str(e)})
                progress.progress(i/max(1,len(selected_models)))
            results_df=pd.DataFrame(results).sort_values("R2_Test",ascending=False,na_position="last").reset_index(drop=True); st.session_state.results=results_df; st.session_state.trained_models=fitted; st.session_state.predictions=prediction_data; st.session_state.convergence=convergence_data; st.session_state.base_actual_predicted=pd.DataFrame(base_actual_pred); st.session_state.model_config={"ann_hidden":int(ann_h),"n_inputs":n_in,"anfis":hybrid_cfg["ANFIS"],"pso":hybrid_cfg["PSO-ANN"],"ga":hybrid_cfg["GA-ANN"],"gwo":hybrid_cfg["GWO-ANN"],"mpa":hybrid_cfg["MPA-ANN"],"ooa":hybrid_cfg["OOA-ANN"],"cgf":hybrid_cfg["CGF-ANN"]}; status.success("Training completed successfully."); st.dataframe(results_df,use_container_width=True)

# CGF-ANN Nh=2–30 research sweep (independent of the normal model run).
if cgf_sweep:
    st.markdown("### CGF-ANN Hidden-Neuron Analysis (Nh = 2–30)")
    run_cgf_sweep=st.button("▶ RUN CGF-ANN Nh=2–30",type="secondary",use_container_width=True,key="run_cgf_sweep")
    if run_cgf_sweep:
        if split_mode=="Auto Split":
            if uploaded_file is None:
                st.error("Please upload an Excel (.xlsx) file before running the CGF sweep."); st.stop()
            df_s=read_excel_sheet(uploaded_file,sheet_name); Xs=df_s.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; ys=pd.to_numeric(df_s.iloc[:,-1],errors="coerce").values; valid_s=np.isfinite(Xs).all(axis=1)&np.isfinite(ys); Xs=Xs[valid_s]; ys=ys[valid_s]
            Xtr_s,Xte_s,ytr_s,yte_s=train_test_split(Xs,ys,test_size=test_percent/100,random_state=int(seed),shuffle=shuffle=="Yes")
        else:
            if train_file is None or test_file is None:
                st.error("Upload both training and testing Excel files before running the CGF sweep."); st.stop()
            tr_s=read_excel_sheet(train_file,train_sheet); te_s=read_excel_sheet(test_file,test_sheet); Xtr_s=tr_s.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; ytr_s=pd.to_numeric(tr_s.iloc[:,-1],errors="coerce").values; Xte_s=te_s.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; yte_s=pd.to_numeric(te_s.iloc[:,-1],errors="coerce").values
        scaler_cls={"StandardScaler":StandardScaler,"MinMaxScaler":MinMaxScaler,"RobustScaler":RobustScaler,"None":None}[scaler_name]
        if scaler_cls is not None:
            sc_s=scaler_cls(); Xtr_s=sc_s.fit_transform(Xtr_s); Xte_s=sc_s.transform(Xte_s)
        sweep_cfg={"hidden":10,"iterations":int(cgf_iter),"step":float(cgf_step),"error_goal":0.0,"lower":float(ann_low),"upper":float(ann_high),"seed":int(seed)}
        prog_s=st.progress(0); status_s=st.empty(); rows_s=[]; hist_s={}; pred_s=[]
        for idx,nh in enumerate(range(2,31),1):
            status_s.info(f"Running CGF-ANN with Nh={nh} ({idx}/29)")
            cfg_s=dict(sweep_cfg); cfg_s["hidden"]=nh
            model_s,hist=cgf_ann_fit(Xtr_s,ytr_s,cfg_s)
            ptr_s=ann_predict(model_s,Xtr_s,Xtr_s.shape[1],nh); pte_s=ann_predict(model_s,Xte_s,Xtr_s.shape[1],nh)
            rows_s.append({"Nh":nh,"Model":"CGF-ANN","R2_Train":r2_score(ytr_s,ptr_s),"RMSE_Train":np.sqrt(mean_squared_error(ytr_s,ptr_s)),"MAE_Train":mean_absolute_error(ytr_s,ptr_s),"R2_Test":r2_score(yte_s,pte_s),"RMSE_Test":np.sqrt(mean_squared_error(yte_s,pte_s)),"MAE_Test":mean_absolute_error(yte_s,pte_s)})
            for phase, actuals, preds in (("TR", ytr_s, ptr_s), ("TS", yte_s, pte_s)):
                for sample, (actual, pred) in enumerate(zip(actuals, preds), start=1):
                    pred_s.append({"Nh":nh,"Model":"CGF-ANN","Dataset":phase,"Sample":sample,"Actual":float(actual),"Predicted":float(pred),"Residual":float(actual-pred)})
            hist_s[nh]=hist; prog_s.progress(idx/29)
        cgf_sweep_df=pd.DataFrame(rows_s)
        st.session_state.cgf_sweep_results=cgf_sweep_df; st.session_state.cgf_sweep_history=hist_s; st.session_state.cgf_sweep_predictions=pd.DataFrame(pred_s)
        status_s.success("CGF-ANN Nh=2–30 sweep completed.")
    if "cgf_sweep_results" in st.session_state:
        sdf=st.session_state.cgf_sweep_results
        st.dataframe(sdf,use_container_width=True)
        st.line_chart(sdf.set_index("Nh")[["R2_Train","R2_Test"]])
        sbuf=io.BytesIO()
        with pd.ExcelWriter(sbuf,engine="openpyxl") as sw:
            sdf.to_excel(sw,index=False,sheet_name="CGF-ANN_Nh_2-30")
            if "cgf_sweep_predictions" in st.session_state:
                st.session_state.cgf_sweep_predictions.to_excel(sw,index=False,sheet_name="CGF_Nh_2-30_Actual_Pred")
            for nh,hdf in st.session_state.get("cgf_sweep_history",{}).items(): hdf.to_excel(sw,index=False,sheet_name=f"CGF_Nh_{nh}_Conv")
        if "cgf_sweep_predictions" in st.session_state:
            st.markdown("#### CGF-ANN Nh=2–30 Actual vs Predicted")
            cp=st.session_state.cgf_sweep_predictions
            csel=st.selectbox("Select CGF-ANN hidden neurons (Nh)", sorted(cp["Nh"].unique().tolist()), key="cgf_pred_nh")
            cphase=st.radio("Dataset", ["TR","TS"], horizontal=True, key="cgf_pred_phase")
            st.dataframe(cp[(cp["Nh"]==csel)&(cp["Dataset"]==cphase)][["Sample","Actual","Predicted","Residual"]], use_container_width=True, height=300)
        st.download_button("⬇ DOWNLOAD CGF Nh=2–30 EXCEL",sbuf.getvalue(),"CGF-ANN_Nh_2-30_Results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="download_cgf_sweep")


# ANN-MPA / ANN-OOA population and iteration analysis for 8-17-1 and 8-26-1.
st.markdown("### ANN-MPA / ANN-OOA Population–Iteration Analysis")
st.caption("Runs architectures 8-17-1 and 8-26-1 for population sizes 10–100 at 500 and 1000 iterations. Results are exported to Excel.")
run_mpa_ooa_sweep=st.button("▶ RUN ANN-MPA + ANN-OOA POPULATION ANALYSIS",type="secondary",use_container_width=True,key="run_mpa_ooa_sweep")
if run_mpa_ooa_sweep:
    if split_mode=="Auto Split":
        if uploaded_file is None:
            st.error("Please upload an Excel (.xlsx) file before running the MPA/OOA analysis."); st.stop()
        df_mo=read_excel_sheet(uploaded_file,sheet_name)
        Xm=df_mo.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; ym=pd.to_numeric(df_mo.iloc[:,-1],errors="coerce").values
        valid_m=np.isfinite(Xm).all(axis=1)&np.isfinite(ym); Xm=Xm[valid_m]; ym=ym[valid_m]
        Xtr_m,Xte_m,ytr_m,yte_m=train_test_split(Xm,ym,test_size=test_percent/100,random_state=int(seed),shuffle=shuffle=="Yes")
    else:
        if train_file is None or test_file is None:
            st.error("Upload both training and testing Excel files before running the MPA/OOA analysis."); st.stop()
        tr_m=read_excel_sheet(train_file,train_sheet); te_m=read_excel_sheet(test_file,test_sheet)
        Xtr_m=tr_m.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; ytr_m=pd.to_numeric(tr_m.iloc[:,-1],errors="coerce").values
        Xte_m=te_m.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").values; yte_m=pd.to_numeric(te_m.iloc[:,-1],errors="coerce").values
    if Xtr_m.shape[1] != 8:
        st.error(f"The requested architectures 8-17-1 and 8-26-1 require exactly 8 input parameters. Your dataset has {Xtr_m.shape[1]} predictors.")
    else:
        scaler_cls_m={"StandardScaler":StandardScaler,"MinMaxScaler":MinMaxScaler,"RobustScaler":RobustScaler,"None":None}[scaler_name]
        if scaler_cls_m is not None:
            sc_m=scaler_cls_m(); Xtr_m=sc_m.fit_transform(Xtr_m); Xte_m=sc_m.transform(Xte_m)
        base_m={"hidden":17,"population":10,"iterations":500,"lower":float(ann_low),"upper":float(ann_high),"seed":int(seed)}
        total=2*2*2*10
        prog_m=st.progress(0); status_m=st.empty(); counter=0
        # Run each experiment with live progress.
        rows_m=[]; hist_m={}; pred_m=[]
        for arch in ((8,17,1),(8,27,1)):
            for algorithm in ("MPA","OOA"):
                for iterations in (500,1000):
                    for population in range(10,101,10):
                        counter+=1; status_m.info(f"Running ANN-{algorithm} {arch[0]}-{arch[1]}-{arch[2]} | population={population} | iterations={iterations} ({counter}/{total})")
                        cfg=dict(base_m); cfg.update(hidden=arch[1],population=population,iterations=iterations)
                        cfg["seed"]=int(seed)+arch[1]*100000+population*100+iterations+(0 if algorithm=="MPA" else 5000000)
                        weights,hist=optimized_ann_fit(algorithm,Xtr_m,ytr_m,cfg)
                        ptr=ann_predict(weights,Xtr_m,8,arch[1]); pte=ann_predict(weights,Xte_m,8,arch[1])
                        rows_m.append({"Architecture":f"8-{arch[1]}-1","Model":f"ANN-{algorithm}","Population":population,"Iterations":iterations,"R2_Train":float(r2_score(ytr_m,ptr)),"RMSE_Train":float(np.sqrt(mean_squared_error(ytr_m,ptr))),"MAE_Train":float(mean_absolute_error(ytr_m,ptr)),"R2_Test":float(r2_score(yte_m,pte)),"RMSE_Test":float(np.sqrt(mean_squared_error(yte_m,pte))),"MAE_Test":float(mean_absolute_error(yte_m,pte))})
                        for phase, actuals, preds in (("TR", ytr_m, ptr), ("TS", yte_m, pte)):
                            for idx, (actual, pred) in enumerate(zip(actuals, preds), start=1):
                                pred_m.append({"Architecture":f"8-{arch[1]}-1","Model":f"ANN-{algorithm}","Population":population,"Iterations":iterations,"Dataset":phase,"Sample":idx,"Actual":float(actual),"Predicted":float(pred),"Residual":float(actual-pred)})
                        hist_m[f"{algorithm}_8-{arch[1]}-1_{iterations}_{population}"]=hist
                        prog_m.progress(counter/total)
        mo_df=pd.DataFrame(rows_m)
        st.session_state.mpa_ooa_sweep_results=mo_df; st.session_state.mpa_ooa_sweep_history=hist_m; st.session_state.mpa_ooa_sweep_predictions=pd.DataFrame(pred_m)
        status_m.success("ANN-MPA / ANN-OOA population analysis completed.")

if "mpa_ooa_sweep_results" in st.session_state:
    mo_df=st.session_state.mpa_ooa_sweep_results
    st.subheader("Population analysis results")
    st.dataframe(mo_df,use_container_width=True)
    if "mpa_ooa_sweep_predictions" in st.session_state:
        st.markdown("#### Actual vs Predicted results")
        pmo=st.session_state.mpa_ooa_sweep_predictions
        psel=st.selectbox("Select architecture / optimizer / population / iterations", sorted(pmo.apply(lambda r: f"{r['Architecture']} | {r['Model']} | Population {int(r['Population'])} | Iterations {int(r['Iterations'])}",axis=1).unique()), key="mo_pred_select")
        pp=pmo[pmo.apply(lambda r: f"{r['Architecture']} | {r['Model']} | Population {int(r['Population'])} | Iterations {int(r['Iterations'])}"==psel,axis=1)]
        trp=pp[pp["Dataset"]=="TR"]; tsp=pp[pp["Dataset"]=="TS"]
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Training (TR)**"); st.dataframe(trp[["Sample","Actual","Predicted","Residual"]],use_container_width=True,height=260)
        with c2:
            st.markdown("**Testing (TS)**"); st.dataframe(tsp[["Sample","Actual","Predicted","Residual"]],use_container_width=True,height=260)
    st.markdown("#### R² versus population")
    for arch in ("8-17-1","8-26-1"):
        for alg in ("ANN-MPA","ANN-OOA"):
            for itv in (500,1000):
                q=mo_df[(mo_df["Architecture"]==arch)&(mo_df["Model"]==alg)&(mo_df["Iterations"]==itv)].sort_values("Population")
                if not q.empty:
                    st.markdown(f"**{alg} — {arch} — {itv} iterations**")
                    st.line_chart(q.set_index("Population")[["R2_Train","R2_Test"]])
    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf,engine="openpyxl") as xw:
        mo_df.to_excel(xw,index=False,sheet_name="All_Results")
        if "mpa_ooa_sweep_predictions" in st.session_state:
            st.session_state.mpa_ooa_sweep_predictions.to_excel(xw,index=False,sheet_name="Actual_Predicted_All")
        for arch in ("8-17-1","8-26-1"):
            for alg in ("ANN-MPA","ANN-OOA"):
                for itv in (500,1000):
                    q=mo_df[(mo_df["Architecture"]==arch)&(mo_df["Model"]==alg)&(mo_df["Iterations"]==itv)].sort_values("Population")
                    q.to_excel(xw,index=False,sheet_name=f"{alg.replace('-','_')}_{arch}_{itv}"[:31])
    st.download_button("⬇ DOWNLOAD ANN-MPA + ANN-OOA POPULATION EXCEL",xbuf.getvalue(),"ANN_MPA_OOA_8-17-1_8-26-1_Population_Results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="download_mpa_ooa_population")

with tab2:
    st.markdown('<div class="section-title">RESULTS & ANALYSIS</div>',unsafe_allow_html=True); results=st.session_state.results
    if results is None: st.info("Run Training from the Workbench tab first.")
    else:
        valid=results.dropna(subset=["R2_Test"])
        if len(valid):
            best=valid.iloc[0]; a,b,c,d=st.columns(4); a.metric("Best Model",best["Model"]); b.metric("R² Test",f'{best["R2_Test"]:.4f}'); c.metric("RMSE Test",f'{best["RMSE_Test"]:.4f}'); d.metric("MAE Test",f'{best["MAE_Test"]:.4f}')
        result_tab1,result_tab2,result_tab3=st.tabs(["Training Result","Testing Result","Iteration / Epoch Analysis"])
        with result_tab1:
            training_cols=[c for c in results.columns if c in ["Model","R2_Train","RMSE_Train","MAE_Train","Time_s","Configuration","Error"]]
            st.subheader("Training Result")
            st.dataframe(results[training_cols],use_container_width=True)
        with result_tab2:
            testing_cols=[c for c in results.columns if c in ["Model","R2_Test","RMSE_Test","MAE_Test","Time_s","Configuration","Error"]]
            st.subheader("Testing Result")
            st.dataframe(results[testing_cols],use_container_width=True)
            if "best_hyperparams" in st.session_state and st.session_state.best_hyperparams is not None:
                st.subheader("Best Hyperparameters — 16 Base Models")
                st.dataframe(st.session_state.best_hyperparams,use_container_width=True)
            if "base_actual_predicted" in st.session_state and not st.session_state.base_actual_predicted.empty:
                st.subheader("Actual vs Predicted — 16 Base Models")
                st.dataframe(st.session_state.base_actual_predicted,use_container_width=True,height=350)
            st.subheader("Observed vs Predicted")
            if st.session_state.trained_models:
                sm=st.selectbox("Select Model",list(st.session_state.trained_models.keys()))
                pdf=st.session_state.predictions[sm]
                st.line_chart(pdf[["Observed","Predicted"]])
                st.download_button("⬇ DOWNLOAD PREDICTIONS",pdf.to_csv(index=False),f"{sm}_Predictions.csv","text/csv")
        with result_tab3:
            conv=st.session_state.get("convergence",{})
            if not conv:
                st.info("Iteration/epoch history is available for ANFIS, PSO-ANN, GA-ANN, GWO-ANN, MPA-ANN, OOA-ANN, ANFIS-MPA and ANFIS-OOA. Select one of these models to view RMSE/MAE convergence.")
            else:
                cm=st.selectbox("Select optimization model",list(conv.keys()),key="conv_model")
                hdf=conv[cm]
                metric_options=["RMSE"] if cm in ["ANFIS","MPA-ANN","OOA-ANN","ANFIS-MPA","ANFIS-OOA"] else ["RMSE","MAE"]
                metric=st.radio("Plot metric",metric_options,horizontal=True,key="conv_metric")
                st.line_chart(hdf.set_index("Iteration")[[metric]])
                st.dataframe(hdf,use_container_width=True,height=300)
                st.download_button("⬇ DOWNLOAD CONVERGENCE DATA",hdf.to_csv(index=False),f"{cm}_Convergence.csv","text/csv")

        # Excel workbook with dedicated Training Result and Testing Result sheets.
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            train_export=results[[c for c in results.columns if c in ["Model","R2_Train","RMSE_Train","MAE_Train","Time_s","Configuration","Error"]]].copy()
            test_export=results[[c for c in results.columns if c in ["Model","R2_Test","RMSE_Test","MAE_Test","Time_s","Configuration","Error"]]].copy()
            train_export.to_excel(w,index=False,sheet_name="Training Result")
            test_export.to_excel(w,index=False,sheet_name="Testing Result")
            for mn,hdf in st.session_state.get("convergence",{}).items(): hdf.to_excel(w,index=False,sheet_name=f"{mn}_Convergence"[:31])
            if "cgf_sweep_results" in st.session_state:
                st.session_state.cgf_sweep_results.to_excel(w,index=False,sheet_name="CGF-ANN_Nh_2-30")
                if "cgf_sweep_predictions" in st.session_state:
                    st.session_state.cgf_sweep_predictions.to_excel(w,index=False,sheet_name="CGF_Nh_2-30_Actual_Pred")
            if "mpa_ooa_sweep_results" in st.session_state:
                mo_export=st.session_state.mpa_ooa_sweep_results.copy()
                mo_export.to_excel(w,index=False,sheet_name="MPA_OOA_Pop_All")
                if "mpa_ooa_sweep_predictions" in st.session_state:
                    st.session_state.mpa_ooa_sweep_predictions.to_excel(w,index=False,sheet_name="MPA_OOA_Actual_Pred")
                for arch in ("8-17-1","8-26-1"):
                    for alg in ("ANN-MPA","ANN-OOA"):
                        for itv in (500,1000):
                            q=mo_export[(mo_export["Architecture"]==arch)&(mo_export["Model"]==alg)&(mo_export["Iterations"]==itv)].sort_values("Population")
                            q.to_excel(w,index=False,sheet_name=f"{alg.replace('-','_')}_{arch}_{itv}"[:31])
            if "best_hyperparams" in st.session_state and st.session_state.best_hyperparams is not None:
                st.session_state.best_hyperparams.to_excel(w,index=False,sheet_name="Best_Hyperparameters_16")
            if "base_actual_predicted" in st.session_state and not st.session_state.base_actual_predicted.empty:
                st.session_state.base_actual_predicted.to_excel(w,index=False,sheet_name="Actual_Predicted_16")
            for mn,pd_ in st.session_state.predictions.items(): pd_.to_excel(w,index=False,sheet_name=f"{mn}_Pred"[:31])
        st.download_button("⬇ DOWNLOAD MODEL_RESULTS.XLSX",buf.getvalue(),"MLRweb_Model_Results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab3:
    st.markdown('<div class="section-title">CODE GENERATOR</div>',unsafe_allow_html=True)
    if st.session_state.results is None: st.info("Run Training first to generate the configuration code.")
    else:
        cfg=st.session_state.get("model_config",{}); st.code(f'''# MLRweb — Auto Generated Configuration\n# Developed by Dr. Furquan Ahmad\n\nimport pandas as pd\n\nSHEET_NAME_INPUT = "T"\n\n# Last column = target; preceding columns = predictors\ndf = pd.read_excel("your_file.xlsx", sheet_name=SHEET_NAME_INPUT)\nX = df.iloc[:, :-1].values\ny = df.iloc[:, -1].values\n\n# Dynamic ANN/ANFIS settings used in the last run\nN_INPUTS = {cfg.get("n_inputs", "auto")}\nNH = {cfg.get("ann_hidden", "user-defined")}\nANFIS_CLUSTERS = {cfg.get("anfis",{}).get("n_clusters", "user-defined")}\nANFIS_M = {cfg.get("anfis",{}).get("m", "user-defined")}\nANFIS_FCM_ITERATIONS = {cfg.get("anfis",{}).get("fcm_iterations", "user-defined")}\nANFIS_EPOCHS = {cfg.get("anfis",{}).get("epochs", "user-defined")}\nPSO_SWARM = {cfg.get("pso",{}).get("swarm", "user-defined")}\nPSO_ITERATIONS = {cfg.get("pso",{}).get("iterations", "user-defined")}\nGA_POPULATION = {cfg.get("ga",{}).get("population", "user-defined")}\nGA_GENERATIONS = {cfg.get("ga",{}).get("generations", "user-defined")}\nGWO_WOLVES = {cfg.get("gwo",{}).get("wolves", "user-defined")}\nGWO_ITERATIONS = {cfg.get("gwo",{}).get("iterations", "user-defined")}\n''',language="python")

st.divider(); st.caption("MLRweb — ML Workbench | Developed by Dr. Furquan Ahmad | Excel Sheet Name Edition | Dynamic ANN/ANFIS Research Models")
