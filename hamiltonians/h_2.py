
# this function return a unique hamiltonian, futur implementations should allow building your proper hamiltonian
# TODO mhh: allow the use and calculation of multiple hamiltonians
def get_h2_hamiltonian_terms() -> list[tuple[float, str]]:
    hamiltonian_terms = [
        (-0.24274280046588792, 'IIZI'),
        (-0.24274280046588792, 'IIIZ'),
        (-0.04207898539364302, 'IIII'),
        (0.17771287502681438, 'ZIII'),
        (0.1777128750268144,  'IZII'),
        (0.12293305045316086, 'ZIZI'),
        (0.12293305045316086, 'IZIZ'),
        (0.16768319431887935, 'ZIIZ'),
        (0.16768319431887935, 'IZZI'),
        (0.1705973836507714,  'ZZII'),
        (0.1762764072240811,  'IIZZ'),
        (-0.044750143865718496, 'YYXX'),
        (-0.044750143865718496, 'XXYY'),
        (0.044750143865718496, 'YXXY'),
        (0.044750143865718496, 'XYYX')
    ]
    return hamiltonian_terms

def get_h2_tfi_hamiltonian_terms() -> list[tuple[str, float]]:
    h = 0.25
    hamiltonian_terms = [
        ("ZZIIII", -1),
        ("IZZIII", -1),
        ("IIZZII", -1),
        ("IIIZZI", -1),
        ("IIIIZZ", -1),
        ("XIIIII", -h),
        ("IXIIII", -h),
        ("IIXIII", -h),
        ("IIIXII", -h),
        ("IIIIXI", -h),
        ("IIIIIX", -h)]
    return hamiltonian_terms