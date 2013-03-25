$(document).ready(function()
{
    if (!mturk_isassigned())
    {
        mturk_acceptfirst();
    }
    else
    {
        mturk_showstatistics();
    }
});
