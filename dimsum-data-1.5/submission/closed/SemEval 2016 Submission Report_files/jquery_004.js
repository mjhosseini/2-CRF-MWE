function setToolTip() {
    jQuery( document ).tooltip({
      items: "[tagtitle], [title]",
      content: function() {
        var element = jQuery( this );
        if ( element.is( "[tagtitle]" ) ) {
          return jQuery("#" + element.attr( "tagtitle" )).html();
        }
        if ( element.is( "[title]" ) ) {
          return element.attr( "title" );
        }
      }
    });
}

function setSpecialedit() {
  jQuery(document).ready(function() {

    jQuery('.helpergreek').specialedit([
					'<span title="Start <sup>Superscript</sup>"><small><small>&lt;sup&gt;</small></small></span>',
					'<span title="End <sup>Superscript</sup>"><small><small>&lt;/sup&gt;</small></small></span>',
					'<span title="Start <sub>Subscript</sub>"><small><small>&lt;sub&gt;</small></small></span>',
					'<span title="End <sub>Subscript</sub>"><small><small>&lt;/sub&gt;</small></small></span>',
					'<span title="Alpha">&Alpha;</span>',
					'<span title="Beta">&Beta;</span>',
					'<span title="Gamma">&Gamma;</span>',
					'<span title="Delta">&Delta;</span>',
					'<span title="Epsilon">&Epsilon;</span>',
					'<span title="Zeta">&Zeta;</span>',
					'<span title="Eta">&Eta;</span>',
					'<span title="Theta">&Theta;</span>',
					'<span title="Iota">&Iota;</span>',
					'<span title="Kappa">&Kappa;</span>',
					'<span title="Lambda">&Lambda;</span>',
					'<span title="Mu">&Mu;</span>',
					'<span title="Nu">&Nu;</span>',
					'<span title="Xi">&Xi;</span>',
					'<span title="Omicron">&Omicron;</span>',
					'<span title="Pi">&Pi;</span>',
					'<span title="Rho">&Rho;</span>',
					'<span title="Sigma">&Sigma;</span>',
					'<span title="Tau">&Tau;</span>',
					'<span title="Upsilon">&Upsilon;</span>',
					'<span title="Phi">&Phi;</span>',
					'<span title="Chi">&Chi;</span>',
					'<span title="Omega">&Omega;</span>',
					'<span title="alpha">&alpha;</span>',
					'<span title="beta">&beta;</span>',
					'<span title="gamma">&gamma;</span>',
					'<span title="delta">&delta;</span>',
					'<span title="epsilon">&epsilon;</span>',
					'<span title="zeta">&zeta;</span>',
					'<span title="eta">&eta;</span>',
					'<span title="theta">&theta;</span>',
					'<span title="iota">&iota;</span>',
					'<span title="kappa">&kappa;</span>',
					'<span title="lambda">&lambda;</span>',
					'<span title="mu">&mu;</span>',
					'<span title="nu">&nu;</span>',
					'<span title="xi">&xi;</span>',
					'<span title="omicron">&omicron;</span>',
					'<span title="pi">&pi;</span>',
					'<span title="rho">&rho;</span>',
					'<span title="sigmaf">&sigmaf;</span>',
					'<span title="sigma">&sigma;</span>',
					'<span title="tau">&tau;</span>',
					'<span title="upsilon">&upsilon;</span>',
					'<span title="phi">&phi;</span>',
					'<span title="chi">&chi;</span>',
					'<span title="psi">&psi;</span>',
					'<span title="omega">&omega;</span>',
					'<span title="thetasym">&thetasym;</span>',
					'<span title="upsih">&upsih;</span>',
					'<span title="piv">&piv;</span>',
                                      
                                       ], 
				       { lineNumber: 4 });
      
  });
}


// Active tab
function setTabMagic() {
    //  To return to the same tab on refresh and redirect.
    //
    //  Documentation
    //      http://api.jqueryui.com/tabs/#option-active
    //      http://api.jqueryui.com/tabs/#event-activate
    //      http://balaarjunan.wordpress.com/2010/11/10/html5-session-storage-key-things-to-consider/
    //
    //  Define friendly index name
    var index = 'key';
    //  Define friendly data store name
    var dataStore = window.sessionStorage;
    //  Start magic!
    try {
        // getter: Fetch previous value
        var oldIndex = dataStore.getItem(index);
    } catch(e) {
        // getter: Always default to first tab in error state
        var oldIndex = 0;
    }
    jQuery('#tabs').tabs({
        // The zero-based index of the panel that is active (open)
        active : oldIndex,
        // Triggered after a tab has been activated
        activate : function( event, ui ){
            //  Get future value
            var newIndex = ui.newTab.parent().children().index(ui.newTab);
            //  Set future value
            dataStore.setItem( index, newIndex ) 
        }
    });

    var hash = jQuery.trim( window.location.hash );
    if (hash) {
      var index = jQuery('#tabs a[href="'+hash+'"]').parent().index();
      jQuery("#tabs").tabs("option", "active", index);
    }
}
