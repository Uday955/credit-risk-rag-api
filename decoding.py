# decoding.py

sub_grades_sorted = sorted(['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                             'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                             'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5',
                             'G1','G2','G3','G4','G5'])
sub_grade_decode = {i: sg for i, sg in enumerate(sub_grades_sorted)}

grade_decode = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

def build_state_mapping(X_test, X_test_fair):
    """Build a mapping from frequency-encoded state values to actual state names."""
    return dict(zip(X_test['addr_state'].round(6), X_test_fair['addr_state']))

def decode_feature_value(feature_name, value, state_mapping=None):
    """Convert an encoded feature value back to a human-readable form."""
    if feature_name == 'sub_grade':
        return sub_grade_decode.get(int(value), value)
    elif feature_name == 'grade':
        return grade_decode.get(int(value), value)
    elif feature_name == 'term':
        return f"{int(value)} months"
    elif feature_name == 'addr_state' and state_mapping is not None:
        return state_mapping.get(round(value, 6), value)
    elif feature_name.startswith('home_ownership_') or feature_name.startswith('purpose_') or feature_name.startswith('verification_status_'):
        return "Yes" if value == 1 or value == True else "No"
    else:
        return value