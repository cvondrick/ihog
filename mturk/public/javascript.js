$(document).ready(function()
{
    var showing = 0;

    if (!mturk_isassigned())
    {
        mturk_acceptfirst();
    }
    else
    {
        mturk_showstatistics();
    }

    var parameters = mturk_parameters();
    if (!parameters["id"]) 
    {
        $("body").html("Missing ID query string.");
        return;
    }

    var marks = {};

    server_request("getjob", [parameters["id"]], function(data) {
        var urls = [];
        for (var i = 0; i < data["windows"].length; i++)
        {
            urls.push("/server/getwindow/" + data["windows"][i]);
        }
        preload(urls, function(p) { $("#debug").html(p); } );

        $("#nextim").click(function() {
            if (marks[showing] == undefined)
            {
                alert("Please make a choice before proceeding.");
                return;
            }
            if (showing == data["windows"].length-1) return;
            $("#window" + showing).hide();
            showing++;
            update();
        });
        $("#previm").click(function() {
            if (showing == 0) return;
            $("#window" + showing).hide();
            showing--;
            update();
        });

        $("#doesnotcontain").click(function() {
            marks[showing] = -1;
        });
        $("#doescontain").click(function() {
            marks[showing] = 1;
        });

        $(window).keypress(function(e) {
            if (e.which == 99) {
                $("#doescontain").click();
                $("#nextim").click();
            }
            if (e.which == 110) {
                $("#doesnotcontain").click();
                $("#nextim").click();
            }
        });

        function showwindow(i)
        {
            $("#windows").html("<img src='/server/getwindow/" + data["windows"][i] + "'>");
        }

        function update()
        {
            showwindow(showing);
            $("#status").html("Showing image " + (showing+1) + " of " + data["windows"].length);

            if (marks[showing] == 1)
            {
                $("#doescontain").attr("checked", true);
            }
            else if (marks[showing] == -1)
            {
                $("#doesnotcontain").attr("checked", true);
            }
            else
            {
                $("#doesnotcontain").attr("checked", false);
                $("#doescontain").attr("checked", false);
            }
        }

        update();
    });
});
