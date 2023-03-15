
import numpy as np
import tensorflow           as  tf

##########################################################
# %%
# train data generator 
##########################################################
class data_generator(tf.keras.utils.Sequence):
    
    def __init__(self, kdata_all, csm, mask_trn1, mask_trn2, mask_lss1, mask_lss2 ):
        
        # inputs 
        self.kdata      =   kdata_all
        self.mask_trn1  =   mask_trn1
        self.mask_trn2  =   mask_trn2
        self.mask_lss1  =   mask_lss1
        self.mask_lss2  =   mask_lss2
        self.csm        =   csm     
        
        # vars
        self.n_batch    =   kdata_all.shape[0]       
        self.num_split  =   mask_trn1.shape[0]   
        self.n_slc      =   csm.shape[0] 
        self.on_epoch_end()

    def __len__(self):                
        # diff directions * slices * splits
        return int(self.num_split)

    def on_epoch_end(self):        
        self.indexes        =   np.arange(self.num_split )
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __getitem__(self, index):      
        indexes     = self.indexes[index]
        # Generate data
        C, K_trn1, K_trn2, K_lss1, K_lss2, m_trn1, m_trn2, m_lss1, m_lss2 = self.__data_generation(indexes)        
        return [C, K_trn1, K_trn2, K_lss1, K_lss2, m_trn1, m_trn2, m_lss1, m_lss2], [np.zeros(1,)]

    def __data_generation(self, indexes):
        
        # indexes
        I_slc   =   indexes%self.n_slc
        I_msk   =   indexes
        I_bat   =   indexes%int(self.n_batch)
                
        # data selection
        K1      =   np.copy(self.kdata[I_bat:I_bat+1,:,:,:,0])
        K2      =   np.copy(self.kdata[I_bat:I_bat+1,:,:,:,1])
        C       =   np.copy(self.csm[I_slc:I_slc+1,])
        m_trn1  =   np.copy(self.mask_trn1[I_msk:I_msk+1,])
        m_trn2  =   np.copy(self.mask_trn2[I_msk:I_msk+1,])
        m_lss1  =   np.copy(self.mask_lss1[I_msk:I_msk+1,])
        m_lss2  =   np.copy(self.mask_lss2[I_msk:I_msk+1,])
        
        # taking mask
        K_trn1  =   K1 * np.expand_dims(m_trn1, axis=1)
        K_trn2  =   K2 * np.expand_dims(m_trn2, axis=1)
        K_lss1  =   K1 * np.expand_dims(m_lss1, axis=1)
        K_lss2  =   K2 * np.expand_dims(m_lss2, axis=1)

        return C, K_trn1, K_trn2, K_lss1, K_lss2, m_trn1, m_trn2, m_lss1, m_lss2