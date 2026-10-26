r_for(){  # "$1|$2" = dataset|backbone -> "<final_r> <decay_interval>"; unknown -> rc 1 (cell skipped)
    # GSAT paper datasets, GSAT's own schedule (GSAT_upstream_6812f5c/src/configs): mutag GIN 0.5/10,
    # PNA 0.5/5; ogbg_mol GIN 0.7/20, PNA 0.7/10. GCN/GAT/SAGE (no GSAT config) take GIN's schedule.
    case "$1|$2" in
        "mutag|GAT") echo "0.5 10" ;;
        "mutag|GCN") echo "0.5 10" ;;
        "mutag|GIN") echo "0.5 10" ;;
        "mutag|PNA") echo "0.5 5" ;;
        "mutag|SAGE") echo "0.5 10" ;;
        "ogbg-molhiv|GAT") echo "0.7 20" ;;
        "ogbg-molhiv|GCN") echo "0.7 20" ;;
        "ogbg-molhiv|GIN") echo "0.7 20" ;;
        "ogbg-molhiv|PNA") echo "0.7 10" ;;
        "ogbg-molhiv|SAGE") echo "0.7 20" ;;
        "ogbg-molbace|GAT") echo "0.7 20" ;;
        "ogbg-molbace|GCN") echo "0.7 20" ;;
        "ogbg-molbace|GIN") echo "0.7 20" ;;
        "ogbg-molbace|PNA") echo "0.7 10" ;;
        "ogbg-molbace|SAGE") echo "0.7 20" ;;
        "ogbg-molbbbp|GAT") echo "0.7 20" ;;
        "ogbg-molbbbp|GCN") echo "0.7 20" ;;
        "ogbg-molbbbp|GIN") echo "0.7 20" ;;
        "ogbg-molbbbp|PNA") echo "0.7 10" ;;
        "ogbg-molbbbp|SAGE") echo "0.7 20" ;;
        "ogbg-moltox21|GAT") echo "0.7 20" ;;
        "ogbg-moltox21|GCN") echo "0.7 20" ;;
        "ogbg-moltox21|GIN") echo "0.7 20" ;;
        "ogbg-moltox21|PNA") echo "0.7 10" ;;
        "ogbg-moltox21|SAGE") echo "0.7 20" ;;
        "ogbg-molclintox|GAT") echo "0.7 20" ;;
        "ogbg-molclintox|GCN") echo "0.7 20" ;;
        "ogbg-molclintox|GIN") echo "0.7 20" ;;
        "ogbg-molclintox|PNA") echo "0.7 10" ;;
        "ogbg-molclintox|SAGE") echo "0.7 20" ;;
        *) return 1 ;;
    esac
}
