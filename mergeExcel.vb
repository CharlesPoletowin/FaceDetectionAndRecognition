Sub Merge()
'
' Merge 宏
'
 Application.Goto Reference:="Merge"

    Application.DisplayAlerts = False

    For i = [A65536].End(3).Row To 2 Step -1

        If Cells(i - 1, 4) = Cells(i, 4) Then

            Range(Cells(i - 1, 4), Cells(i, 4)).Merge

        End If

    Next

    Application.DisplayAlerts = True
'
End Sub
