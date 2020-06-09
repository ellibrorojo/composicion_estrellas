# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:54:00 2020

@author: Javier
"""

import utils

def generate_topics(tokens):
    
    wsw1_pos =   {
                'name':'Funcionamiento (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # MUY POSITIVO
                        {
                        'syn0': ['works_great',
                                 'works_flawlessly',
                                 'works_perfectly',
                                 'worked_perfectly',
                                 'works_well'],
                        'syn1': [],
                        'syn2': [],
                        'nots': ['not']
                        }
                        # POSITIVO
                        ,{
                        'syn0': ['works_fine',
                                 'good_job',
                                 'working_properly',
                                 'works_ok',
                                 'work_properly',
                                 'serves_purpose'],
                        'syn1': [],
                        'syn2': [],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['works'],
                        'syn2': ['expected'],
                        'nots': ['not']
                        }
                        # BAJO CONSUMO
                        ,{
                        'syn0': [],
                        'syn1': ['power_consumption'],
                        'syn2': ['low'],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw1_neg =   {
                'name':'Funcionamiento (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # GENÉRICO, NO FUNCIONA DEBIDAMENTE
                        {
                        'syn0': ['not_work'],
                        'syn1': ['working_properly',
                                 'work_properly'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['faulty',
                                 'defective',
                                 'fluke'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # PROBLEMAS DE ESTABILIDAD
                        ,{
                        'syn0': ['unstable',
                                 'stability_issues'],
                        'syn1': ['stable'],
                        'syn2': ['not'],
                        'nots': []
                        },
                        {
                        'syn0': [],
                        'syn1': ['suddenly_stopped',
                                 'stopped'],
                        'syn2': ['suddenly',
                                 'working'],
                        'nots': []
                        },
                        {
                        'syn0': ['unreliable',
                                 'works_intermittently',
                                 'poor_reliability'],
                        'syn1': ['reliable'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        # DEJÓ DE FUNCIONAR O COMENZÓ A FUNCIONAR MAL
                        ,{
                        'syn0': [],
                        'syn1': ['quit_working',
                                 'stopped_working',
                                 'stopped_working',
                                 'stop_working',
                                 'quits_working',
                                 'quitted_working'],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['not_durable'],
                        'syn1': ['durable'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['broke'],
                        'syn2': ['right_away'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['within_week',
                                 'within_weeks',
                                 'week',
                                 'weeks'],
                        'syn2': ['stopped_working',
                                 'started',
                                 'broke',
                                 'quits_working',
                                 'quitted_working'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['obsolescence',
                                 'short_life',
                                 'never_worked'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # ALTO CONSUMO
                        ,{
                        'syn0': ['drains_battery',
                                 'eat_batteries',
                                 'eats_batteries',
                                 'eating_batteries'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['power_consumption'],
                        'syn2': ['high'],
                        'nots': []
                        },
                        ]
                }
            }
            
    
    wsw2_pos =   {
                'name':'Características (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # GENÉRICO APARIENCIA
                        #{
                        #'syn0': busca_tokens(tokens, ['looks']),
                        #'syn1': [],
                        #'syn2': [],
                        #'nots': []
                        #}
                        # BUENA CALIDAD DE PIEZAS MECÁNICAS Y COMPONENTES
                        {
                        'syn0': ['excellent_build',
                                 'sturdy'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['well_made',
                                 'quite_good',
                                 'well_designed',
                                 'excellent_product',
                                 'high_quality'],
                        'syn1': ['build_quality',
                                 'image_quality'],
                        'syn2': ['good',
                                 'nice',
                                 'high'],
                        'nots': []
                        },
                        {
                        'syn0': ['excellent_product',
                                 'excellent_quality',
                                 'excellent_results',
                                 'excellent_sound'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['sounds_great'],
                        'syn1': ['sound_quality',
                                 'audio_quality'],
                        'syn2': ['good',
                                 'great',
                                 'excellent'],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['picture_quality',
                                 'image_quality'],
                        'syn2': ['good',
                                 'great',
                                 'excellent'],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['battery_life'],
                        'syn2': ['good',
                                 'great',
                                 'long',
                                 'excellent'],
                        'nots': ['not']
                        }
                        # BUEN ASPECTO
                        ,{
                        'syn0': ['nice_looking'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # TAMAÑO ADECUADO
                        ,{
                        'syn0': ['fit_perfectly'],
                        'syn1': ['fit',
                                 'fits'],
                        'syn2': ['good',
                                 'perfectly',
                                 'right'],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': ['perfect_size'],
                        'syn1': ['size',
                                 'length'],
                        'syn2': ['perfect'],
                        'nots': []
                        }
                        ]
                }
            }               
    
    wsw2_neg =   {
                'name':'Características (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # GENÉRICO APARIENCIA
                        #{
                        #'syn0': busca_tokens(tokens, ['looks']),
                        #'syn1': [],
                        #'syn2': [],
                        #'nots': []
                        #}
                        # MALA CALIDAD DE PIEZAS MECÁNICAS Y COMPONENTES
                        {
                        'syn0': utils.busca_tokens(tokens, ['defective']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['broke_apart',
                                 'fell_apart',
                                 'fall_apart',
                                 'falling_apart',
                                 'cheaply_made',
                                 'cheap_plastic',
                                 'poor_quality',
                                 'bit_flimsy',
                                 'defective_product',
                                 'poorly_constructed'],
                        'syn1': ['melted'],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['feels_cheap',
                                 'felt_cheap',
                                 'cheap_feel',
                                 'poorly_made',
                                 'cheap_plastic',
                                 'cheap_plastics',
                                 'cheap_cardboard',
                                 'cheap_feeling',
                                 'cheap_construction',
                                 'cheap_materials',
                                 'cheap_material'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['build_quality'],
                        'syn2': ['bad'],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': ['broke_within',
                                 'easily_broken',
                                 'breaks_easily',
                                 'breaks_easy',
                                 'breaks_every',
                                 'breaks_quickly',
                                 'broke_quickly',
                                 'broke_shortly'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['sound_quality',
                                 'audio_quality'],
                        'syn2': ['poor',
                                 'bad'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['picture_quality',
                                 'image_quality'],
                        'syn2': ['poor',
                                 'bad'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['battery_life'],
                        'syn2': ['short'],
                        'nots': []
                        }
                        # DISEÑO PRESENTA FALLOS
                        ,{
                        'syn0': ['design_flaw',
                                 'design_defect',
                                 'design_weakness',
                                 'flawed_design',
                                 'poor_design'],
                        'syn1': ['serious_design'],
                        'syn2': ['flaw',
                                 'flaws'],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['drawbacks']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['drawback']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # TAMAÑO INADECUADO
                        ,{
                        'syn0': [],
                        'syn1': ['fit',
                                 'fits'],
                        'syn2': ['not',
                                 'no'],
                        'nots': []
                        },
                        {
                        'syn0': [],
                        'syn1': ['big',
                                 'small',
                                 'long',
                                 'short'],
                        'syn2': ['too',
                                 'not_enough',
                                 'way_too'],
                        'nots': []
                        },
                        {
                        'syn0': [],
                        'syn1': ['size',
                                 'length'],
                        'syn2': ['wrong'],
                        'nots': ['no']
                        },
                        {
                        'syn0': ['too_short',
                                 'too_long',
                                 'not_fit',
                                 'too_small',
                                 'too_big',
                                 'too_large'],
                        'syn1': ['long_enough'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        ]
                }
            }               
    
    wsw3_pos =  {
                'name':'Comodidad/Sencillez (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # PUESTA EN FUNCIONAMIENTO SENCILLA O RÁPIDA
                        {
                        'syn0': ['easy_install',
                                 'easy_installation',
                                 'easy_setup',
                                 'installed_easily',
                                 'quick_installation',
                                 'plug_play'],
                        'syn1': ['plug'],
                        'syn2': ['play'],
                        'nots': []
                        #'nots': ['not', 'no']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['installation',
                                 'install',
                                 'setup',
                                 'installed'],
                        'syn2': ['easy',
                                 'fast',
                                 'quick',
                                 'easily',
                                 'smooth',
                                 'breeze'],
                        'nots': []
                        #'nots': ['not', 'no']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['installed'],
                        'syn2': ['easily',
                                 'quickly'],
                        'nots': []
                        #'nots': ['not', 'no']
                        }
                        #,{
                        #'syn0': busca_tokens(tokens, ['install', 'installation', 'installed', 'setup']),
                        #'syn1': [],
                        #'syn2': [],
                        #'nots': []
                        #}
                        ,{
                        'syn0': ['plug_play'],
                        'syn1': ['plug'],
                        'syn2': ['play'],
                        'nots': []
                        }
                        # USO DEL MANUAL DE INSTRUCCIONES
                        ,{
                        #'syn0': ['followed_instructions', 'written_instructions', 'written_documentation', 'instruction_book', 'instruction_booklet', 'instruction_manual', 'instruction_manuals', 'instruction_sheet', 'instructions_included', 'instructions_say'],
                        'syn0': utils.busca_tokens(tokens, ['instructions']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['instruction']),
                        #'syn1': ['instructions'],
                        #'syn2': ['understand', 'unintelligible'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw3_neg =  {
                'name':'Comodidad/Sencillez (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # PUESTA EN FUNCIONAMIENTO COMPLICADA O LENTA
                        #{
                        #'syn0': busca_tokens(tokens, ['install', 'installation', 'installed', 'setup']),
                        #'syn1': [],
                        #'syn2': [],
                        #'nots': []
                        #}
                        {
                        'syn0': [],
                        'syn1': ['plug_play'],
                        'syn2': ['no',
                                 'not'],
                        'nots': []
                        }
                        # PROBLEMAS CON EL MANUAL DE INSTRUCCIONES
                        ,{
                        'syn0': ['indecipherable_instructions',
                                 'confusing_instructions',
                                 'unclear_instructions',
                                 'instructions_stink',
                                 'chinglish_instructions',
                                 'meager_instructions',
                                 'incomplete_instructions',
                                 'scant_instructions',
                                 'engrish_instructions',
                                 'clearer_instructions',
                                 'conflicting_instructions',
                                 'sparse_instructions',
                                 'skimpy_instructions',
                                 'cryptic_instructions',
                                 'inadequate_instructions',
                                 'vague_instructions',
                                 'no_instructions',
                                 'instructions_suck']
                        ,'syn1': ['instructions']
                        ,'syn2': ['understand',
                                  'unintelligible']
                        ,'nots': []
                        }
                        ,{
                        'syn0': ['poor_instruction',
                                 'cryptic_instruction'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # PRODUCTO DE DIFÍCIL USO, INCÓMODOS O CON ELEMENTOS DESAGRADABLES
                        ,{
                        'syn0': [],
                        'syn1': ['user_friendly'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['not_recognize'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['tricky',
                                 'figure_out',
                                 'figured_out',
                                 'figuring_out'],
                        'syn1': ['trying'],
                        'syn2': ['work'],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['annoying']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['uncomfortable']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }   
    
    wsw4 =  {
                'name':'Producto incorrecto'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # GENÉRICO. SE MENCIONA LA DESCRIPCION
                        #{
                        #'syn0': busca_tokens(tokens, ['description']),
                        #'syn1': [],
                        #'syn2': [],
                        #'nots': []
                        #}
                        # SE ENVÍA UN PRODUCTO EQUIVOCADO
                        {
                        'syn0': ['specifically_ordered',
                                 'shipped_wrong'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # EL PRODUCTO NO ES EXACTAMENTE COMO SE DESCRIBE
                        ,{
                        'syn0': [],
                        'syn1': ['advertised'],
                        'syn2': ['exactly'],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['different_than'],
                        'syn2': ['advertised'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['description_implied',
                                 'description_incorrect',
                                 'description_misleads',
                                 'description_neglects',
                                 'description_lies',
                                 'description_boasts',
                                 'description_incomplete',
                                 'deceptive_description',
                                 'description_states',
                                 'description_indicates',
                                 'inaccurate_description',
                                 'false_description',
                                 'description_stating',
                                 'description_suggests',
                                 'incomplete_description',
                                 'description_claims',
                                 'description_inaccurate',
                                 'description_implies',
                                 'improper_description',
                                 'wrong_description',
                                 'phony_description',
                                 'description_omits',
                                 'description_specifies',
                                 'description_indicated',
                                 'description_emphasizes',
                                 'description_said',
                                 'misleading_description',
                                 'description_lists',
                                 'description_stated',
                                 'incorrect_description'],
                        'syn1': ['description'],
                        'syn2': ['incorrect'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['totally_different',
                                 'entirely_different'],
                        'syn2': ['recieved',
                                 'described',
                                 'pictured',
                                 'ordered'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['pictured',
                                 'advertised',
                                 'described'],
                        'syn2': ['not'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['meet_expectation',
                                 'meet_expectations',
                                 'meeting_expectations',
                                 'meets_expectations'],
                        'syn1': [],
                        'syn2': [],
                        'nots': ['not']
                        }
                        ]
                }
            }
    
    wsw5 =  {
                'name':'Referencia a opiniones'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': utils.busca_tokens(tokens, ['review']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['reviews']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['reviewer']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['reviewers']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }   
    
    wsw6_pos =  {
                'name':'Envío/entrega/packaging (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # PAQUETE Y PACKAGING
                        {
                        'syn0': ['fancy_packaging',
                                 'glitzy_packaging',
                                 'sealed_packaging',
                                 'recyclable_packaging',
                                 'free_packaging'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['nicely_packed',
                                 'blister_packed',
                                 'securely_packed',
                                 'packed_nicely',
                                 'neatly_packed',
                                 'safely_packed'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,
                        {
                        'syn0': ['sealed_package',
                                 'tidy_package'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # ENVÍO RÁPIDO
                        ,{
                        'syn0': ['timely_shipping',
                                 'speedy_shipping',
                                 'overnight_shipping',
                                 'ships_quickly',
                                 'shipped_fast',
                                 'shipped_quickly',
                                 'shipped_promptly'],
                        'syn1': ['shipping',
                                 'delivery'],
                        'syn2': ['fast',
                                 'quick',
                                 'good'],
                        'nots': []
                        }
                        ,{
                        'syn0': ['shipped_immediately',
                                 'fast_delivery',
                                 'fast_service',
                                 'fast_ship',
                                 'fast_shipping',
                                 'quick_delivery'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['arrived_timely',
                                 'arrived_earlier',
                                 'arrived_sooner',
                                 'arrived_promptly'],
                        'syn1': ['arrived'],
                        'syn2': ['fast',
                                 'quickly'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['shipment_arrived'],
                        'syn2': ['short_time'],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw6_neg =  {
                'name':'Envío/entrega/packaging (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        # PAQUETE RECIBIDO EN MALAS CONDICIONES
                        {
                        'syn0': ['shipping_damage'],
                        'syn1': ['broken',
                                 'damaged'],
                        'syn2': ['box',
                                 'package'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['package',
                                 'box'],
                        'syn2': ['missing'],
                        'nots': []
                        }
                        # GENÉRICO DE PAQUETE Y PACKAGING
                        ,{
                        'syn0': ['excessive_packaging',
                                 'stupid_packaging',
                                 'poor_packaging',
                                 'lousy_packaging',
                                 'wasteful_packaging'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['densely_packed',
                                 'jam_packed',
                                 'poorly_packed',
                                 'improperly_packed',
                                 'packed_tightly',
                                 'loosely_packed',
                                 'tightly_packed'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,
                        {
                        'syn0': ['incomplete_package'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        # ENVÍO LENTO
                        ,{
                        'syn0': [],
                        'syn1': ['shipping',
                                 'delivery'],
                        'syn2': ['slow'],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw7_pos =  {
                'name':'Devolución/Postventa (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': [],
                        'syn1': ['help_desk',
                                 'customer_support',
                                 'customer_service',
                                 'tech_support',
                                 'call_customer',
                                 'called_customer'],
                        'syn2': ['great',
                                 'good',
                                 'excellent'
                                 'helpful'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['warranty_service',
                                 'manufacturer_warranty',
                                 'warranty'],
                        'syn2': ['great',
                                 'good',
                                 'excellent'
                                 'nice'],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw7_neg =  {
                'name':'Devolución/Postventa (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': [],
                        'syn1': ['help_desk',
                                 'customer_support',
                                 'customer_service',
                                 'tech_support',
                                 'call_customer',
                                 'called_customer'],
                        'syn2': ['poor'],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['warranty_service',
                                 'manufacturer_warranty',
                                 'warranty'],
                        'syn2': ['poor',
                                 'terrible',
                                 'horrible'],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw8 =  {
                'name':'Precio'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': utils.busca_tokens(tokens, ['price']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['priced']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['dollar']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['dollars']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['expensive']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['inexpensive']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['cheap', 'ridiculously_cheap', 'incredibly_cheap', 'super_cheap', 'kinda_cheap', 'relatively_cheap'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }
    
    wsw9_neg = {
                'name':'Sentimiento (NEG)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': ['not_buy', 
                                 'not_bother', 
                                 'buyer_beware', 
                                 'never_buy', 
                                 'not_waste', 
                                 'not_wast', 
                                 'throw_away', 
                                 'cannot_recommend', 
                                 'wasted_money'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['return']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': utils.busca_tokens(tokens, ['returning']),
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['sent_back',
                                 'send_back',
                                 'sending_back',
                                 'ship_back'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ]
                }
            }       
    
    wsw9_pos = {
                'name':'Sentimiento (POS)'
                ,'wordset': 
                {
                    'ands': [],
                    'ors' :
                        [
                        {
                        'syn0': ['highly_recommend', 'definitely_recommend', 'no_complaints'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['great_value', 'good_value', 'good_deal', 'well_worth', 'totally_worth', 'definately_worth'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['wise_choice', 'good_choice', 'smart_choice', 'excellent_choice', 'best_choice'],
                        'syn1': [],
                        'syn2': [],
                        'nots': ['not']
                        }
                        ,{
                        'syn0': ['worth_every'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['excellent_product'],
                        'syn1': [],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': ['exceeds_expectations', 'greatly_exceeds', 'exceeds_expectation', 'far_exceeds'],
                        'syn1': ['pleasantly_surprised'],
                        'syn2': [],
                        'nots': []
                        }
                        ,{
                        'syn0': [],
                        'syn1': ['than_expected'],
                        'syn2': ['much_better', 'much', 'better'],
                        'nots': []
                        }
                        ]
                }
            }
            
    topics = [wsw1_pos
              , wsw1_neg
              , wsw2_pos
              , wsw2_neg
              , wsw3_pos
              , wsw3_neg
              , wsw4
              , wsw5
              , wsw6_pos
              , wsw6_neg
              , wsw7_pos
              , wsw7_neg
              , wsw8
              , wsw9_pos
              , wsw9_neg]

    master_dict = {}
    for topic in topics:
        master_dict[topic['name']]=topic
          
    return master_dict

def get_topic_by_name(topics, name):
    return topics[name]
        