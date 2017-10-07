
dnow = new Date()

function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}

function nextPage() {
    
    var aid=$('[name="assignmentId"]').val()
    var dat = {
        'c':$('[name="c"]').val(),
        's':$('[name="s"]').val(),
        'assignmentId':aid,
        'workerId':$('[name="workerId"]').val(),
        'time': (new Date() - dnow),
        'clustering':$('[name="clustering"]').val()
    }

    for (var i = 0; i < """ + str(sizex * sizey) + """; i++) {
        if ($('#i' + i).is(':checked')) {
            dat['i' + i] = true
        }
    }
    if (aid != 'ASSIGNMENT_ID_NOT_AVAILABLE') {
        $.ajax({type:'POST', url:'/submit', data: JSON.stringify(dat), contentType:'application/json'}).done(function() {
            document.getElementById('form1').submit()
            //location.reload()
        })
    }
}

document.onkeypress = function(e) {
    e = e || window.event;
    if (e.keyCode == 110) { //yes
        nextItem(false);
    } else if (e.keyCode == 109) { // no
        nextItem(true);
    }
}


currItemIndex = 0;
function nextItem(checked) {
    
    $('#i' + currItemIndex).parent().hide();
    $('#i' + currItemIndex).prop('checked', checked);
    currItemIndex++;
    if (currItemIndex < """ + str(sizex*sizey) + """) { 
        $('#i' + currItemIndex).parent().show();
    } else {
        nextPage();
    }
}
