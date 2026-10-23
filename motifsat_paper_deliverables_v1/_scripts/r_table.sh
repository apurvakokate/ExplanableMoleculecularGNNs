r_for(){  # "$1|$2" = dataset|backbone -> "<final_r> <decay_interval>"; unknown -> rc 1 (cell skipped)
    case "$1|$2" in
        "Alkane_Carbonyl_Verified_GT|GAT") echo "0.5 20" ;;
        "Alkane_Carbonyl_Verified_GT|GCN") echo "0.7 20" ;;
        "Alkane_Carbonyl_Verified_GT|GIN") echo "0.5 10" ;;
        "Alkane_Carbonyl_Verified_GT|PNA") echo "0.5 20" ;;
        "Alkane_Carbonyl_Verified_GT|SAGE") echo "0.7 20" ;;
        "BBBP|GAT") echo "0.5 20" ;;
        "BBBP|GCN") echo "0.7 20" ;;
        "BBBP|GIN") echo "0.7 20" ;;
        "BBBP|PNA") echo "0.5 20" ;;
        "BBBP|SAGE") echo "0.5 20" ;;
        "Benzene_Verified_GT|GAT") echo "0.5 10" ;;
        "Benzene_Verified_GT|GCN") echo "0.5 20" ;;
        "Benzene_Verified_GT|GIN") echo "0.7 20" ;;
        "Benzene_Verified_GT|PNA") echo "0.5 20" ;;
        "Benzene_Verified_GT|SAGE") echo "0.5 20" ;;
        "Fluoride_Carbonyl_Verified_GT|GAT") echo "0.5 20" ;;
        "Fluoride_Carbonyl_Verified_GT|GCN") echo "0.5 20" ;;
        "Fluoride_Carbonyl_Verified_GT|GIN") echo "0.7 20" ;;
        "Fluoride_Carbonyl_Verified_GT|PNA") echo "0.5 10" ;;
        "Fluoride_Carbonyl_Verified_GT|SAGE") echo "0.5 20" ;;
        "Lipophilicity|GAT") echo "0.5 20" ;;
        "Lipophilicity|GCN") echo "0.7 20" ;;
        "Lipophilicity|GIN") echo "0.7 20" ;;
        "Lipophilicity|PNA") echo "0.5 20" ;;
        "Lipophilicity|SAGE") echo "0.5 20" ;;
        "Mutagenicity|GAT") echo "0.5 20" ;;
        "Mutagenicity|GCN") echo "0.7 20" ;;
        "Mutagenicity|GIN") echo "0.7 20" ;;
        "Mutagenicity|PNA") echo "0.5 20" ;;
        "Mutagenicity|SAGE") echo "0.5 20" ;;
        "esol|GAT") echo "0.5 20" ;;
        "esol|GCN") echo "0.7 20" ;;
        "esol|GIN") echo "0.7 20" ;;
        "esol|PNA") echo "0.5 20" ;;
        "esol|SAGE") echo "0.5 20" ;;
        "hERG|GAT") echo "0.5 20" ;;
        "hERG|GCN") echo "0.7 20" ;;
        "hERG|GIN") echo "0.7 20" ;;
        "hERG|PNA") echo "0.5 20" ;;
        "hERG|SAGE") echo "0.5 20" ;;
        *) return 1 ;;
    esac
}
