{
    M += $4;
    M2 += $4 * $4;
    count += 1;
}
NR % 200 == 0 {
    total_count += 1;

    avgM += M / count;
    avgM2 += M2 / count;

    X += (avgM2 - avgM * avgM) / $1;

    avgM = 0;
    avgM2 = 0;
    M = 0;
    M2 = 0;
    count = 0;
}
END { print $1, X / total_count; }
