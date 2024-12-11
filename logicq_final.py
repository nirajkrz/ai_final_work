def analyze_desirability(CS, AW, LQ, HM):
    # Compute intermediate values based on the circuit logic
    I1 = CS or AW  # CS ∨ AW
    I2 = not (AW and not LQ)  # ¬(AW ∧ ¬LQ)
    I3 = CS and (not HM or LQ)  # CS ∧ (HM ⇒ LQ)
    I4 = AW and (LQ == (not CS))  # AW ∧ (LQ ⇔ ¬CS)
    I5 = (not I1) or I2  # I1 ⇒ I2
    I6 = (I2 or I3) <= (I4 <= I3)  # (I2 ∨ I3) ⇒ (I4 ⇒ I3)

    # Desirable is I5 ⇒ ¬I6
    desirable = (not I5) or (not I6)  # Equivalent to I5 ⇒ ¬I6
    return desirable


# Generate and print results for all combinations of CS, AW, LQ, HM
print(f"{'CS':<3} {'AW':<3} {'LQ':<3} {'HM':<3} {'Desirable':<10}")
print("-" * 30)

for CS in [0, 1]:
    for AW in [0, 1]:
        for LQ in [0, 1]:
            for HM in [0, 1]:
                result = analyze_desirability(CS, AW, LQ, HM)
                print(f"{CS:<3} {AW:<3} {LQ:<3} {HM:<3} {str(result):<10}")
