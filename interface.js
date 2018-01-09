
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
        'p':$('[name="p"]').val(),
        'assignmentId':aid,
        'workerId':$('[name="workerId"]').val(),
        'time': (new Date() - dnow),
        'clustering':$('[name="clustering"]').val(),
        'responses':[]
    }

    for (var i = 0; i < NUM_IMAGES; i++) {
        if ($('#i' + i).is(':checked')) {
            dat['responses'][i] = 1
        } else {
            dat['responses'][i] = -1
        }
    }
    // if (aid != 'ASSIGNMENT_ID_NOT_AVAILABLE') {
        if (!submitted)
        {
            submitted = true
            $.ajax({type:'POST', url:'/submit', data: JSON.stringify(dat), contentType:'application/json'}).done(function() {
                document.getElementById('form1').submit()
            })
        }
    // }
}

document.onkeypress = function(e) {
    e = e || window.event;
    if (e.keyCode == 110) { //yes
        nextItem(false);
    } else if (e.keyCode == 109) { // no
        nextItem(true);
    }
}

submitted = false;
currItemIndex = 0;
function nextItem(checked) {
    
    $('#i' + currItemIndex).parent().hide();
    $('#i' + currItemIndex).prop('checked', checked);
    currItemIndex++;
    if (currItemIndex < NUM_IMAGES) { 
        $('#i' + currItemIndex).parent().show();
    } else {
        nextPage();
    }
}
