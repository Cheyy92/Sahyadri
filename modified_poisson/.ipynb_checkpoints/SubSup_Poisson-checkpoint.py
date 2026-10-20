import numpy as np

def poiss_dist_AP(lam,Ap=1.0,itp=0,lamU=0,dist='poission',rec=0,max_rec=20):
    '''returns the poission distribution
    dist: poission generate possion distribution with Ap
    if dist= gaussian generate gaussian distribution with mean as lam and sigma=f*lam
    where f is the sqrt(var/20) with var being the variance of poission with mean as lam and given Ap'''
    
    #a universal lam needs to set for iterative modes
    if(lamU==0):
        lamU=lam
    
    if(dist=='gaussian'):
        #get the variance of possion with mean as 20
        #pdist=poiss_dist_AP(20,Ap=Ap,itp=2,lamU=20,dist='poission')
        #mu_t,var_t=mu_var_dist(pdist)
        #fac=var_t/20.0
        gstd=np.sqrt(Ap*lam)
        xx_min=max(0,np.int(lam-5*gstd))
        xx_max=np.int(lam+5*gstd)
        xx=np.arange(xx_min,xx_max)
        gexp=0.5*np.power((xx-lam)/gstd,2)
        fp=np.exp(-gexp)
        fp=fp/np.sum(fp)
        pdist=np.column_stack([xx,fp])
    elif(dist=='poission'):
        lam_p=lam/Ap
        xx=np.arange(0,max(10,6*lam))
        xx_p=xx*Ap

        #factorial=np.append(1,np.cumprod(xx[1:]))
        factorial=np.append(1,np.cumprod(xx[1:].astype('float')))
        fp=1.0*np.power(lam_p,xx)*np.exp(-lam_p)/factorial
        #print(nn,factorial)

        #interpolate this to integer values
        if(itp==0):
            pdist=np.column_stack([xx_p,fp])
        elif(itp==1):
            xx_out=np.arange(0,np.int(xx_p.max()-1))
            fp_out=np.interp(xx_out,xx_p,fp)
            pdist=np.column_stack([xx_out,fp_out])
        elif(itp==2): #itrate to get the correct mean
            xx_out=np.arange(0,max(np.int(xx_p.max()-1),10))
            fp_out=np.interp(xx_out,xx_p,fp,right=0)
            pdist=np.column_stack([xx_out,fp_out])
            mu_t,var_t=mu_var_dist(pdist)
            if(np.abs(mu_t-lamU)>0.015*lamU and rec<max_rec):
                if(mu_t>lamU):
                    fac=0.9
                else:
                    fac=1.05
                #print(lam,mu_t,var_t,fac,lamU)
                pdist=poiss_dist_AP(fac*lam,Ap=Ap,itp=2,lamU=lamU,rec=rec+1)
            #elif(rec>=max_rec):
                #print('maximu recursion:',lam,mu_t,var_t,Ap,lamU)
                
           # pdist=pdist
            
    #normalizing the distribution
    pdist[:,1]=pdist[:,1]/np.sum(pdist[:,1])
    
    return pdist

def mu_var_dist(pdist):
    mu=np.sum(pdist[:,0]*pdist[:,1])/np.sum(pdist[:,1])
    var=np.sum(pdist[:,0]*pdist[:,0]*pdist[:,1])/np.sum(pdist[:,1])
    var=var-np.power(mu,2)
    
    return mu,var


def Generate_ss_pois(lam,Ap=1.0,size=1,plots=1):
    '''Generates random number for a given value of Ap and mean'''
    
    #first get the distribution
    if(lam<20):
        pdist=poiss_dist_AP(lam,Ap=Ap,itp=2)
    else:
        pdist=poiss_dist_AP(lam,Ap=Ap,itp=2,dist='gaussian')
    #now generate the distribution
    #print(pdist,size)
    Nreal=np.random.choice(pdist[:,0].astype('int'),p=pdist[:,1],size=size)
    
    #Gaussian random numbers
    #Nreal=np.random.normal(loc=lam,scale=np.sqrt(lam),size=size)
    #print(pdist.astype('int'))
    if(plots==1): #this compares the two distribution
        import pylab as pl
        pl.plot(pdist[:,0],pdist[:,1],'k-',lw=2)
        hbin=np.linspace(0,Nreal.max()+1,Nreal.max()+2)-0.5
        hbin_mid=0.5*(hbin[1:]+hbin[:-1])
        #print(hbin)
        Nhist,hh=np.histogram(Nreal,bins=hbin,normed=1)
        pl.plot(hbin_mid,Nhist,'o-')
        
    return Nreal


def Get_SubSup_poission_real(meanSat_All,Ap=1.0,dlam=0.5):
    '''Creates the random number with sub/super -possion distribution for given mean and Ap parameter
    This has been tested for Ap=[0.5,2.0] and works well with 1.5% accuracy in mean
    Note that when mean<4 then the Ap scaling is not as disrect on the variance
    but it smaller than the value of AP
    The distribution is generated in a discrete manner with nlam number of discrete plaes'''

    #Now write to Nsat All
    Nsat_All=np.zeros(meanSat_All.size,dtype=int)
    #First remove everything with less than 0.5 meanSat
    ind_sel=meanSat_All>0.5
    meanSat=meanSat_All[ind_sel]

    maxmean=meanSat.max()
    Bsplit_arr=np.array([0.5,2.0,4.0,8.0,25.0,100000])
    dlam_arr=np.array([0.1,0.1,0.1,0.5,0.5])

    Nsat_real=np.zeros(meanSat.size,dtype=int)
    for bb in range(0,Bsplit_arr.size-1):
        ind1=meanSat>=Bsplit_arr[bb]
        ind2=meanSat<Bsplit_arr[bb+1]
        ind_bsel=ind1*ind2
        if(np.sum(ind_bsel)==0 and Bsplit_arr[bb+1]>maxmean):
            break
        elif(np.sum(ind_bsel)==0):
            continue

        lam_min=meanSat[ind_bsel].min()
        lam_max=meanSat[ind_bsel].max()+0.1
        nlam=np.int((lam_max-lam_min)/dlam_arr[bb])

        lam_ed=np.linspace(lam_min,lam_max,nlam+1)
        lam_mid=0.5*(lam_ed[1:]+lam_ed[:-1])
         
        Nsat_bsel=np.zeros(np.sum(ind_bsel),dtype=int)
        for ll in range(0,nlam):
            ind1=meanSat[ind_bsel]>=lam_ed[ll]
            ind2=meanSat[ind_bsel]<lam_ed[ll+1]
            ind_t=ind1*ind2
            if(np.sum(ind_t)==0):
                continue
            #print(np.sum(ind_t))
            Nsat_bsel[ind_t]=Generate_ss_pois(lam_mid[ll],Ap=Ap,size=np.sum(ind_t),plots=0)

            #Nsat_real[ind_t]=Generate_ss_pois(lam_mid[ll],Ap=Ap,size=np.sum(ind_t),plots=0)

        if(np.sum(ind_bsel)>0):
            Nsat_real[ind_bsel]=Nsat_bsel

    Nsat_All[ind_sel]=Nsat_real

    return Nsat_All

