import numpy as np
from functools import reduce

class make_dicts:
    def __init__(self, user_data_dict, user_model_dict,
                 user_reg_dict, user_path_dict, user_misc_dict):
        self.user_data_dict = user_data_dict
        self.user_model_dict = user_model_dict
        self.user_reg_dict = user_reg_dict
        self.user_path_dict = user_path_dict
        self.user_misc_dict = user_misc_dict
        
        # the final dictionaries to be used by the iterative solver
        self.data_dict = self.make_data_dict()
        self.model_dict = self.make_model_dict()
        self.reg_dict = self.make_reg_dict()
        self.path_dict = self.make_path_dict()
        self.misc_dict = self.make_misc_dict()
        
    def make_data_dict(self):
        user_data_dict = self.user_data_dict
        
        # keys provided by the user
        user_data_keys = user_data_dict.keys()
        
        # keys that must be present in the data dictionary provided by user
        mandatory_data_keys = np.array(['data'])
        
        # checking if the mandatory keys are present
        for key in mandatory_data_keys:
            if(key in user_data_keys):
                pass
            else:
                raise ValueError(f"Missing mandatory data key: {key}")
                
        # default data keys for optional parameters
        optional_data_dict = {}
        optional_data_dict['C_d'] = np.diag(np.ones_like(user_data_dict['data']))
        
        # all data keys: user_specified + optional
        all_data_dicts = np.array([user_data_dict, optional_data_dict])
        all_data_keys = reduce(lambda x, y: x.union(y.keys()), all_data_dicts, set())
        
        # making the entire data dictionary including mandatory and extra keys
        data_dict = {}
        for key in all_data_keys:
            if(key in user_data_keys):
                data_dict[f'{key}'] = user_data_dict[f'{key}']
            else:
                data_dict[f'{key}'] = optional_data_dict[f'{key}']
                
        return data_dict
        
        
    def make_model_dict(self):
        user_model_dict = self.user_model_dict
        
        # keys provided by the user                                                       
        user_model_keys = user_model_dict.keys()
        
        # keys that must be present in the data dictionary provided by user
        mandatory_model_keys = np.array(['G'])
        
        # checking if the mandatory keys are present                                      
        for key in mandatory_model_keys:
            if(key in user_model_keys):
                pass
            else:
                raise ValueError(f"Missing mandatory model key: {key}")
                
        # default data keys for optional parameters                                       
        optional_model_dict = {}
        optional_model_dict['F'] = np.zeros_like(self.user_data_dict['data'])
        
        # all data keys: user_specified + optional                                        
        all_model_dicts = np.array([user_model_dict, optional_model_dict])
        all_model_keys = reduce(lambda x, y: x.union(y.keys()), all_model_dicts, set())
        
        # making the entire data dictionary including mandatory and extra keys            
        model_dict = {}
        for key in all_model_keys:
            if(key in user_model_keys):
                model_dict[f'{key}'] = user_model_dict[f'{key}']
            else:
                model_dict[f'{key}'] = optional_model_dict[f'{key}']
                
        return model_dict

    def make_reg_dict(self):
        user_reg_dict = self.user_reg_dict

        # keys provided by the user                                                           
        user_reg_keys = user_reg_dict.keys()

        # keys that must be present in the data dictionary provided by user                   
        mandatory_reg_keys = np.array(['mu', 'D'])

        # checking if the mandatory keys are present                                          
        for key in mandatory_reg_keys:
            if(key in user_reg_keys):
                pass
            else:
                raise ValueError(f"Missing mandatory regularization key: {key}")

        # default data keys for optional parameters                                           
        optional_reg_dict = {}

        # all data keys: user_specified + optional                                            
        all_reg_dicts = np.array([user_reg_dict, optional_reg_dict])
        all_reg_keys = reduce(lambda x, y: x.union(y.keys()), all_reg_dicts, set())

        # making the entire data dictionary including mandatory and extra keys                
        reg_dict = {}
        for key in all_reg_keys:
            if(key in user_reg_keys):
                reg_dict[f'{key}'] = user_reg_dict[f'{key}']
            else:
                reg_dict[f'{key}'] = optional_reg_dict[f'{key}']

        return reg_dict

    
    def make_path_dict(self):
        user_path_dict = self.user_path_dict

        # keys provided by the user                                                           
        user_path_keys = user_path_dict.keys()

        # keys that must be present in the data dictionary provided by user                   
        mandatory_path_keys = np.array([])

        # checking if the mandatory keys are present                                          
        for key in mandatory_path_keys:
            if(key in user_path_keys):
                pass
            else:
                raise ValueError(f"Missing mandatory path key: {key}")

        # default data keys for optional parameters                                           
        optional_path_dict = {}
        optional_path_dict['outdir'] = '.'

        # all data keys: user_specified + optional                                            
        all_path_dicts = np.array([user_path_dict, optional_path_dict])
        all_path_keys = reduce(lambda x, y: x.union(y.keys()), all_path_dicts, set())

        # making the entire data dictionary including mandatory and extra keys                
        path_dict = {}
        for key in all_path_keys:
            if(key in user_path_keys):
                path_dict[f'{key}'] = user_path_dict[f'{key}']
            else:
                path_dict[f'{key}'] = optional_path_dict[f'{key}']

        return path_dict


    def make_misc_dict(self):
        user_misc_dict = self.user_misc_dict

        # keys provided by the user                                                           
        user_misc_keys = user_misc_dict.keys()

        # making the entire data dictionary including mandatory and extra keys                
        misc_dict = {}
        for key in user_misc_keys:
            misc_dict[f'{key}'] = user_misc_dict[f'{key}']

        return misc_dict
