$times = @()
for ($i = 1; $i -le 10; $i++) {
    $result = Measure-Command { .\fastawc.exe -l -w -m big.txt }
    $times += $result.TotalSeconds
    Write-Host "Run ${i}: $($result.TotalSeconds) sec"
}

$stats = $times | Measure-Object -Average -Minimum -Maximum
Write-Host "Average time: $($stats.Average) sec"
Write-Host "Min time:     $($stats.Minimum) sec"
Write-Host "Max time:     $($stats.Maximum) sec"
