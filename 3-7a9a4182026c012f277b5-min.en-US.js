(window["wpJsonpExtractCss"]=window["wpJsonpExtractCss"]||[]).push([[3],{"./src/main/webapp/frontend/packages/enums/IndexType.js":function(e,t){var a={STACKED:"stacked",GRID:"grid",FULL_URL:"full-url"};e.exports=a},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/GoogleSearchPreview.js":function(e,t,a){"use strict";var n=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireDefault.js");var r=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireWildcard.js");Object.defineProperty(t,"__esModule",{value:true});t.default=void 0;var d=r(a("./common/temp/node_modules/react/index.js"));var c=a("./common/temp/node_modules/@sqs/core-components/build/lib/index.js");var p=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/externals.js");var m=n(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/hooks/useSeoData.js"));var f=n(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/GoogleSearchPreview.less"));var s=function e(t){var a=t.value,n=t.isContentItem;var r=(0,m.default)(a,n),s=r.baseUrl,o=r.isHomepage,i=r.previewTitle;var l=a.description,u=a.urlId;return d.default.createElement(d.Fragment,null,d.default.createElement(c.Label,{context:true,label:(0,p.t)("Search Results Preview")}),d.default.createElement("div",{className:f.default.wrapper},d.default.createElement(c.SEOPreview,{description:l||"",placeholderDescription:o?(0,p.t)("This description will be automatically generated by search engines. To override that description, enter a description in website SEO settings."):(0,p.t)("This description will be automatically generated by search engines. To override that description, enter one below."),title:i,link:o?s:"".concat(s,"/").concat(u)})))};var o=s;t.default=o;e.exports=t.default},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/GoogleSearchPreview.less":function(e,t,a){e.exports={wrapper:"GoogleSearchPreview-wrapper-1L6B9"}},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/ImageField.js":function(e,t,a){"use strict";var n=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireWildcard.js");var r=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireDefault.js");Object.defineProperty(t,"__esModule",{value:true});t.default=void 0;var E=r(a("./common/temp/node_modules/@babel/runtime/helpers/extends.js"));var d=r(a("./common/temp/node_modules/@babel/runtime/regenerator/index.js"));var s=r(a("./common/temp/node_modules/@babel/runtime/helpers/asyncToGenerator.js"));var o=r(a("./common/temp/node_modules/@babel/runtime/helpers/classCallCheck.js"));var i=r(a("./common/temp/node_modules/@babel/runtime/helpers/createClass.js"));var l=r(a("./common/temp/node_modules/@babel/runtime/helpers/assertThisInitialized.js"));var u=r(a("./common/temp/node_modules/@babel/runtime/helpers/inherits.js"));var c=r(a("./common/temp/node_modules/@babel/runtime/helpers/possibleConstructorReturn.js"));var p=r(a("./common/temp/node_modules/@babel/runtime/helpers/getPrototypeOf.js"));var m=r(a("./common/temp/node_modules/@babel/runtime/helpers/defineProperty.js"));var f=r(a("./common/temp/node_modules/lodash/pick.js"));var v=r(a("./common/temp/node_modules/lodash/isObject.js"));var A=r(a("./common/temp/node_modules/lodash/isFunction.js"));var g=r(a("./common/temp/node_modules/lodash/isEmpty.js"));var h=r(a("./common/temp/node_modules/lodash/debounce.js"));var S=r(a("./common/temp/node_modules/lodash/omit.js"));var b=r(a("./common/temp/node_modules/prop-types/index.js"));var y=n(a("./common/temp/node_modules/react/index.js"));var O=a("./common/temp/node_modules/@sqs/core-components/build/lib/index.js");var j=a("./common/temp/node_modules/@sqs/rosetta-primitives/build/lib/index.js");var I=a("./common/temp/node_modules/@sqs/rosetta-elements/build/lib/index.js");var P=a("./src/main/webapp/frontend/packages/enums/RecordType.js");var w=r(a("./src/main/webapp/frontend/packages/enums/WorkflowStates.js"));var T=a("./src/main/webapp/frontend/packages/enums/LicensedAssetSource.js");var F=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/externals.js");var k=r(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/ImageField.less"));function C(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(t);e&&(n=n.filter(function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}));a.push.apply(a,n)}return a}function _(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?C(Object(a),true).forEach(function(e){(0,m.default)(t,e,a[e])}):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):C(Object(a)).forEach(function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))})}return t}function x(r){var s=L();return function e(){var t=(0,p.default)(r),a;if(s){var n=(0,p.default)(this).constructor;a=Reflect.construct(t,arguments,n)}else a=t.apply(this,arguments);return(0,c.default)(this,a)}}function L(){if("undefined"===typeof Reflect||!Reflect.construct)return false;if(Reflect.construct.sham)return false;if("function"===typeof Proxy)return true;try{Date.prototype.toString.call(Reflect.construct(Date,[],function(){}));return true}catch(e){return false}}var U={UPLOAD_DONE:"uploadDone",UPLOAD_CANCELED:"uploadCanceled",PROCESSING_DONE:"processingDone"};var R={mediaFocalPoint:{x:.5,y:.5},licensedAssetPreview:null,assetUrl:"",licensedAssetId:null};var N=function e(){return y.default.createElement(j.Box,{px:2},y.default.createElement(I.Divider,null))};var D=function(e){(0,u.default)(C,e);var n=x(C);function C(){var p;(0,o.default)(this,C);for(var e=arguments.length,t=new Array(e),a=0;a<e;a++)t[a]=arguments[a];p=n.call.apply(n,[this].concat(t));(0,m.default)((0,l.default)(p),"state",_({},R,{imageEditorOpen:false,showLicensedAssetPicker:!p.props.value&&p.props.licensedAssetDestination,purchaseModalOpen:false,licensedAssetType:null,licensedAssetUrl:null}));(0,m.default)((0,l.default)(p),"getImageId",function(){var e;var t=p.props.value;if(!(0,v.default)(t))return t;return null===t||void 0===t?void 0:null===(e=t.image)||void 0===e?void 0:e.id});(0,m.default)((0,l.default)(p),"updateStateForNewContentItem",(0,s.default)(d.default.mark(function e(){var a,n,r,s,o,i,l,u;return d.default.wrap(function e(t){while(1)switch(t.prev=t.next){case 0:a=p.props,n=a.focalPointEditable,r=a.focalPointOverride,s=a.imageEditable,o=a.licensedAssetDestination;i=p.getImageId();if(!((n||s||o)&&i)){t.next=8;break}t.next=5;return p.fetchContentItemData(i);case 5:l=t.sent;u=(0,f.default)(l,Object.keys(R));p.setState(_({},u,{mediaFocalPoint:r||u.mediaFocalPoint||R.mediaFocalPoint}));case 8:case"end":return t.stop()}},e)})));(0,m.default)((0,l.default)(p),"fetchContentItemData",function(e){return F.ContentItemAPI.read(e).then(function(e){var t=e.data;return t})});(0,m.default)((0,l.default)(p),"updateFocalPoint",(0,h.default)(function(e){var t=!(arguments.length>1&&void 0!==arguments[1])||arguments[1];var a=p.props,n=a.onFocalPointChange,r=a.focalPointEditable;var s=p.getImageId();if(!s||!r)return;n?n(e):t&&F.ContentItemAPI.updateField({itemId:s,field:"mediaFocalPoint",value:e})},500));(0,m.default)((0,l.default)(p),"getLicensedAssetProps",function(e){var t=F.PurchasedImageStore.getOriginUrlFromId(e)||"";var a=F.PurchasedImageStore.getAssetTypeFromId(e)||null;return{licensedAssetUrl:t,licensedAssetType:a}});(0,m.default)((0,l.default)(p),"handleRemove",function(){var e=p.state,t=e.licensedAssetPreview,a=e.licensedAssetId,n=e.licensedAssetType;t?(0,F.trackInternal)("getty_clear",{assetId:t.assetId,assetSource:T.GETTY}):a&&(0,F.trackInternal)("licensed_asset_remove",{assetId:a,assetSource:n});p.handleChange({})});(0,m.default)((0,l.default)(p),"handleChange",function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};var t=p.props,a=t.onChange,n=t.licensedAssetDestination,r=t.value;var s=p.state,o=s.assetUrl,i=s.mediaFocalPoint;var l=p.getLicensedAssetProps(e.licensedAssetId);var u=(0,g.default)(e);var d=p.getImageId()===e.id;var c=e.mediaFocalPoint!==i;!u&&d&&c?p.updateFocalPoint(e.mediaFocalPoint):d||p.updateFocalPoint({x:.5,y:.5},false);(0,v.default)(r)?a(_({},r,{image:e.id?_({},r.image,{focalPoint:e.mediaFocalPoint},e):{}}),{isSameImage:d}):a(e.id,{isSameImage:d});p.setState(_({},(0,f.default)(e,Object.keys(R)),{},l,{showLicensedAssetPicker:!e.id&&n}));e.assetUrls&&e.assetUrls[0].url!==o&&p.setState({assetUrl:e.assetUrls[0].url})});(0,m.default)((0,l.default)(p),"handleUploadEvent",function(e){var t=p.getImageId();e.action===U.UPLOAD_DONE?p.setState({showLicensedAssetPicker:false}):e.action===U.UPLOAD_CANCELED&&p.props.licensedAssetDestination&&!t&&p.setState({showLicensedAssetPicker:true})});(0,m.default)((0,l.default)(p),"handleLicensedImagePicked",function(e){if(!p.props.uploaderRef.current){console.error("Image Uploader not ready.");return}F.ImagePickerViewActions.finishLoadingImage();F.ImagePickerViewActions.closeImagePicker();if(e.cloneItemId){F.ContentItemAPI.copy(e.cloneItemId,w.default.PUBLISHED,null).then(function(e){var t=e.data;return p.handleChange(t)});return}var t=e.licensedAssetId,a=e.url;p.setState({licensedAssetId:t,licensedAssetUrl:a},function(){p.props.uploaderRef.current.uploadFileFromUrl(_({},p.props.uploadData,{},(0,F.buildUploadPostData)(e,P.IMAGE,F.TRANSIENT)))})});(0,m.default)((0,l.default)(p),"handleSaveImageEditor",function(e){if(!p.props.uploaderRef.current){console.error("Image Uploader not ready.");return}p.setState({imageEditorOpen:false});p.props.uploaderRef.current.uploadFiles([e])});(0,m.default)((0,l.default)(p),"handlePurchase",function(e){if(!p.props.uploaderRef.current){console.error("Image Uploader not ready.");return}var t=p.getImageId();p.props.uploaderRef.current.uploadFileFromUrl((0,F.buildUploadPostData)(e,P.IMAGE,F.TRANSIENT,{replaceItemId:t}));p.setState({licensedAssetPreview:null});p.handleClosePurchaseModal()});(0,m.default)((0,l.default)(p),"handleClickEdit",function(){p.props.onClickEdit?p.props.onClickEdit():p.setState({imageEditorOpen:true})});(0,m.default)((0,l.default)(p),"handleCloseImageEditor",function(){p.setState({imageEditorOpen:false})});(0,m.default)((0,l.default)(p),"handleOpenPurchaseModal",function(){var e=p.state.licensedAssetPreview;p.setState({purchaseModalOpen:true});(0,F.trackInternal)("getty_purchase",{assetId:e.assetId,assetSource:T.GETTY})});(0,m.default)((0,l.default)(p),"handleClosePurchaseModal",function(){p.setState({purchaseModalOpen:false})});(0,m.default)((0,l.default)(p),"renderLeftButton",function(){var e=p.state,t=e.licensedAssetPreview,a=e.assetUrl;if(t)return y.default.createElement(j.Button.Tertiary,{onClick:p.handleOpenPurchaseModal},(0,F.t)("Purchase License"));if(a)return y.default.createElement(j.Button.Tertiary,{onClick:p.handleClickEdit,"data-test":"file-button-edit"},(0,F.t)("Image Editor"));return null});return p}(0,i.default)(C,[{key:"componentDidMount",value:function(){var e=(0,s.default)(d.default.mark(function e(){var a;return d.default.wrap(function e(t){while(1)switch(t.prev=t.next){case 0:t.next=2;return this.updateStateForNewContentItem();case 2:if(!this.state.licensedAssetId){t.next=8;break}a=F.PurchasedImageStore.getImageById(this.state.licensedAssetId);if(a){t.next=7;break}t.next=7;return F.PurchasedImageActions.fetch();case 7:this.setState(this.getLicensedAssetProps(this.state.licensedAssetId));case 8:case"end":return t.stop()}},e,this)}));function t(){return e.apply(this,arguments)}return t}()},{key:"componentDidUpdate",value:function e(t){t.focalPointOverride!==this.props.focalPointOverride&&this.setState({mediaFocalPoint:this.props.focalPointOverride})}},{key:"componentWillUnmount",value:function e(){this.updateFocalPoint.flush()}},{key:"getFocalPoint",value:function e(){var t=this.state.mediaFocalPoint;var a=this.props.invertFocalPoint;var n=t.x,r=t.y;return{x:n,y:a?1-r:r}}},{key:"getPreviewComponent",value:function e(){var n=this;var t=this.props,a=t.height,r=t.focalPointEditable,s=t.invertFocalPoint,o=t.previewKey,i=t.renderPreview;if(i)return i;var l=this.getImageId();var u={value:l,maxHeight:a,dataField:false};var d={focalPoint:this.getFocalPoint(),onFocalPointChange:function e(t,a){return n.handleChange({id:l,mediaFocalPoint:{x:t,y:s?1-a:a}})}};if(r)return function(){return y.default.createElement(O.ImageField,(0,E.default)({key:o},u,d))};return function(){return y.default.createElement(O.ImageField,u)}}},{key:"render",value:function e(){var t;var a=this.state,n=a.imageEditorOpen,r=a.assetUrl,s=a.showLicensedAssetPicker,o=a.licensedAssetPreview,i=a.purchaseModalOpen,l=a.licensedAssetId,u=a.licensedAssetUrl,d=a.licensedAssetType;var c=this.props,p=c.value,m=c.licensedAssetDestination,f=c.imageEditable,v=c.mimeTypes,g=c.height,h=c.errors,b=c.getUploadUrl;var I=this.getImageId();var P=f||o;var w;(0,A.default)(b)&&(w=b(p,l));return y.default.createElement(y.Fragment,null,y.default.createElement(O.FileUpload,(0,E.default)({},(0,S.default)(this.props,Object.keys(C.propTypes)),{uploadUrl:w,ref:this.props.uploaderRef,dispatchValueType:"contentItem",onChange:this.handleChange,onEvent:this.handleUploadEvent,value:I,shouldRenderButtons:!P,height:g,className:k.default.uploadField,mimeTypes:v,fileType:"image",renderPreview:this.getPreviewComponent(),errors:h,"data-test":"image-field"})),I&&P&&y.default.createElement(y.Fragment,null,y.default.createElement("div",{className:k.default.previewButtons},this.renderLeftButton(),y.default.createElement(j.Button.Danger,{onClick:this.handleRemove},(0,F.t)("Remove Image"))),y.default.createElement(N,null)),I&&o&&y.default.createElement(y.Fragment,null,y.default.createElement(O.Context,{type:"label",description:(0,F.t)("This is an unlicensed preview image. To purchase its license, click the Purchase License button above. Watermarks will be removed after purchase")}),y.default.createElement(F.LicensedImagePurchaseModal,{key:"licensed-image-purchase-modal",assetPreview:o,previewUrl:r,isOpen:i,onRequestClose:this.handleClosePurchaseModal,onPurchaseSuccess:this.handlePurchase})),l&&d===T.GETTY&&y.default.createElement(O.Context,{type:"label",description:(0,F.t)("This is a licensed image. Click {link} to review licensing options for print, social, and more.",{link:'<a href="'.concat(u,'" target="_blank">\n              ').concat((0,F.t)("here"),"</a>")})}),s&&y.default.createElement(y.Fragment,null,y.default.createElement(j.Button.Tertiary,{m:2,ml:3,onClick:F.ImagePickerViewActions.openImagePicker,"data-test":"file-button-licensed-image-get"},(0,F.t)("Search For Images")),y.default.createElement(N,null),y.default.createElement(F.LicensedImagePicker,{disableImageReuse:this.props.disableImageReuse,onUnlicensedPreview:this.handleLicensedImagePicked,onLicensedPreview:this.handleLicensedImagePicked,imageDestination:m})),r&&f&&y.default.createElement(F.ImageEditor,{isOpen:n,imageSrc:r,imageFileName:null===p||void 0===p?void 0:null===(t=p.image)||void 0===t?void 0:t.filename,onCancel:this.handleCloseImageEditor,onSave:this.handleSaveImageEditor}))}}]);return C}(y.Component);(0,m.default)(D,"propTypes",{value:b.default.oneOfType([b.default.string,b.default.object]),onChange:b.default.func.isRequired,onClickEdit:b.default.func,onFocalPointChange:b.default.func,licensedAssetDestination:b.default.string,focalPointEditable:b.default.bool,focalPointOverride:b.default.shape({x:b.default.number,y:b.default.number}),imageEditable:b.default.bool,invertFocalPoint:b.default.bool,mimeTypes:b.default.arrayOf(b.default.string),height:b.default.number,errors:b.default.object,collectionId:b.default.string,getUploadUrl:b.default.func,renderPreview:b.default.func,disableImageReuse:b.default.bool});(0,m.default)(D,"defaultProps",{focalPointEditable:true,height:168,imageEditable:true,invertFocalPoint:true,mimeTypes:["image/gif","image/jpeg","image/png"]});var B=y.default.forwardRef(function(e,t){var a=(0,y.useRef)();return y.default.createElement(D,(0,E.default)({},e,{uploaderRef:t||a}))});t.default=B;e.exports=t.default},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/ImageField.less":function(e,t,a){e.exports={previewButtons:"ImageField-previewButtons-2rSSp",uploadField:"ImageField-uploadField-2JOHc"}},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/SocialPreview.js":function(e,t,a){"use strict";var n=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireWildcard.js");var r=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireDefault.js");Object.defineProperty(t,"__esModule",{value:true});t.default=void 0;var I=r(a("./common/temp/node_modules/@babel/runtime/helpers/slicedToArray.js"));var P=n(a("./common/temp/node_modules/react/index.js"));var w=a("./common/temp/node_modules/@sqs/core-components/build/lib/index.js");var C=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/externals.js");var E=r(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/hooks/useSeoData.js"));var A=r(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/SocialPreview.less"));var S=(0,C.t)("This description will be automatically generated depending on the service this link is shared on. To override that description, enter a description on the SEO tab.");var y=function e(t){var a=t.isHomepage,n=t.seoSocialSharingImageId,r=t.socialLogoImageId,s=t.siteLogoId,o=t.thumbnailImage;if(n)return n;if(a&&!r&&s)return s;if(o)return null;return r};var s=function e(t){var n=t.isContentItem,a=t.value;var r=(0,E.default)(a,n),s=r.baseUrl,o=r.isHomepage,i=r.previewTitle;var l=a.description,u=void 0===l?S:l,d=a.seoImageId,c=a.thumbnailImage;var p=C.WebsiteStore.getState();var m=p.get("socialLogoImageId");var f=p.get("logoImageId");var v=(0,P.useState)(null),g=(0,I.default)(v,2),h=g[0],b=g[1];(0,P.useEffect)(function(){var e={isHomepage:o,seoSocialSharingImageId:d,socialLogoImageId:m,siteLogoId:f};var t=y(e);if(!t){if(n&&c&&c.assetUrls){var a=c.assetUrls.length>0?c.assetUrls[0].url:null;b(a)}else b(null);return}C.ContentItemStore.fetchFromServer(t).then(function(e){return b(e?e.get("assetUrl"):null)})},[n,o,f,d,m,c]);return P.default.createElement(P.Fragment,null,P.default.createElement(w.Label,{context:true,label:(0,C.t)("Social Preview")}),P.default.createElement("div",{className:A.default.wrapper},P.default.createElement(w.SocialPreview,{link:s,cover:h,description:u,title:i})))};var o=s;t.default=o;e.exports=t.default},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/customFields/SocialPreview.less":function(e,t,a){e.exports={wrapper:"SocialPreview-wrapper-9OnRl"}},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/constants/IndexTypes.js":function(e,t,a){"use strict";Object.defineProperty(t,"__esModule",{value:true});t.NONE=void 0;var n="none";t.NONE=n},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/hooks/useSeoData.js":function(e,t,a){"use strict";Object.defineProperty(t,"__esModule",{value:true});t.default=r;var n=a("./common/temp/node_modules/react/index.js");var p=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/utils/collectionSettingsUtils.js");var m=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/externals.js");function r(l,u){var d=m.WebsiteSettingsStore.getState();var c=m.WebsiteStore.getState();return(0,n.useMemo)(function(){var e=c.get("baseUrl")||window.location.origin;var t=c.get("siteTitle")||"";var a=(0,p.isHomepage)(l);var n;n=u?d.itemTitleFormat:a?d.homepageTitleFormat:d.collectionTitleFormat;var r=(0,m.updateLegacyFormat)(n);var s=a?l.title:l.seoTitle||l.title;var o={"%s":{exampleText:t},"%p":{exampleText:s||""},"%i":{exampleText:s}};var i=(0,m.getPreviewString)((0,m.buildRegexes)(o),o,r);return{previewTitle:i,baseUrl:e,isHomepage:a}},[l,u,d,c])}e.exports=t.default},"./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/utils/collectionSettingsUtils.js":function(e,t,a){"use strict";var n=a("./common/temp/node_modules/@babel/runtime/helpers/interopRequireDefault.js");Object.defineProperty(t,"__esModule",{value:true});t.getGenericCollectionSettingsTitle=C;t.hasAccessPermissions=E;t.getCollectionMemberAreaId=T;t.isHomepage=G;t.getIsMAHomepage=V;t.getPageLayoutOptions=q;t.containsHomepage=J;t.getCollectionSettingsEditorContext=Q;t.isVariation=void 0;var s=n(a("./common/temp/node_modules/lodash/nth.js"));var r=n(a("./common/temp/node_modules/lodash/findKey.js"));var o=n(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/store/index.js"));var i=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/store/selectors.js");var l=n(a("./src/main/webapp/frontend/packages/enums/CollectionTypes.js"));var u=n(a("./src/main/webapp/frontend/packages/enums/CollectionOrdering.js"));var d=n(a("./src/main/webapp/frontend/packages/enums/AccessPermissions.js"));var c=n(a("./src/main/webapp/frontend/packages/enums/Flag.js"));var p=n(a("./src/main/webapp/frontend/packages/enums/IndexType.js"));var m=n(a("./src/main/webapp/frontend/packages/enums/Features.js"));var f=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/constants/IndexTypes.js");var v=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/components/GroupedList/index.js");var g=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/shared/utils/collectionUtils.js");var h=a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/externals.js");var b=n(a("./src/main/webapp/universal/src/apps/App/screens/ContentBrowser/bundles/products/isProducts.js"));var I=v.pathUtils.forEachItem,P=v.pathUtils.getGroupFromItemPath,w=v.pathUtils.getItemFromPath;function C(e,t){var a=(0,g.getTemplateConfig)(e);return(0,h.t)("{title} Settings",{title:a.title||t})}function E(){return h.AccessPermissionStore.hasPermission(d.default.CONFIG_CHANGE_STRUCTURE)}function A(e){var t=(0,g.getTemplateConfig)(e);if(!t||!E())return false;var a=t.forcePageSize;var n=e.type===l.default.COLLECTION_TYPE_GENERIC;var r=(0,g.getCollectionOrdering)(e,t)===u.default.CHRONOLOGICAL;return!a&&r&&n&&!t.index&&!t.folder}function S(e){var t=(0,g.getTemplateConfig)(e);if(!t)return false;return t.supportsVideoBackgrounds}function y(){if(!E())return false;return true}function O(t){var e,a,n,r,s;var o=null===(e=window.Y)||void 0===e?void 0:null===(a=e.Squarespace)||void 0===a?void 0:null===(n=a.Singletons)||void 0===n?void 0:null===(r=n.TemplateNavigationList)||void 0===r?void 0:r.getMemberAreaNavigation();var i=(null===o||void 0===o?void 0:null===(s=o.get("itemList"))||void 0===s?void 0:s.toJSON())||[];var l=null;var u=function e(t,a,n){var r=t.children||t.items;if(l)return;for(var s=0;s<r.length;s++){var o,i;if(r[s].collectionId===a){l=l||n;return}if(!(null!==(o=r[s].children)&&void 0!==o&&o.length)&&!(null!==(i=r[s].items)&&void 0!==i&&i.length))continue;e(r[s],a,n)}};i.forEach(function(e){(e.children||e.items)&&u(e,t,e.memberAreaId)});return l}function j(n){var r=(0,i.getInitialSiteLayout)(o.default.getState())[h.Constants.memberAreaNavigationName]||{};var e=I(r,function(e,t){if(e.collectionId===n){var a=w(r,t[0]);return a.memberAreaId}});return e}function T(e){if(!h.MemberAreasFeatureUtils.showInPagesPanel())return null;return(0,h.isReactPagesPanelEnabled)()?j(e):O(e)}function F(e){return e.type===l.default.COLLECTION_TYPE_PAGE||e.type===l.default.SPLASH_PAGE}function k(e){var t=(0,g.getTemplateConfig)(e);if(!t)return false;return t.hasSystemBlogCollectionSettings}function _(e){var t=(0,g.getTemplateConfig)(e);if(!t)return false;return(0,b.default)(e)&&("2.0"===t.systemCollectionVersion||t.hasProductQuickView)}function x(e){var t=(0,g.getTemplateConfig)(e);if(!t)return false;return t.relatedItems&&!(0,b.default)(e)}function L(e){if(h.TemplateVersionUtils.isSevenOne())return false;var t=(0,g.getTemplateConfig)(e);var a=(0,g.getCollectionOrdering)(e,t);if(!t)return a===u.default.CALENDAR;return a===u.default.CALENDAR&&!t.forceEventView}function U(e){return E()&&e.type!==l.default.TEMPLATE_PAGE}function R(e){var t=(0,g.getTemplateConfig)(e);var a=(0,g.getCollectionOrdering)(e,t);if(!t||!E)return false;return!t.isIndex&&!t.isFolder&&e.type===l.default.COLLECTION_TYPE_GENERIC&&a===u.default.CHRONOLOGICAL}function N(){var e=(0,i.getTemplateLayouts)(o.default.getState());return!(0,h.isV8)()&&Object.keys(e).length>1}function D(e){var t=h.SiteNavigationStore.getPathToCollection(e.id);var a=(0,s.default)(t,-2);if(!a||!a.typeName&&!a.index)return f.NONE;var n=(0,g.getTemplateConfig)(a);if(!n)return f.NONE;var r=n.index||Object.values(p.default).indexOf(n.indexType)>-1;if(r)return n.indexType;return f.NONE}function B(e){return h.TemplateVersionUtils.isSevenOne()&&(0,g.getCollectionType)(e)===l.default.COLLECTION_TYPE_PAGE}function M(){return h.FeaturesStore.isFeatureGated(m.default.CODE_INJECTION)}function G(e){return e.id===(0,i.getHomepageCollectionId)(o.default.getState())}function H(e,t){return(0,r.default)(e,function(e){return e===t})}function V(e){if(!h.MemberAreasFeatureUtils.showInPagesPanel())return false;var t=e.id||e;var a=h.MemberAreasStore.getMemberAreasHomepages();var n=H(a,t);return n===T(t)}function q(){var e=(0,i.getTemplateLayouts)(o.default.getState());var t={enum:[],enumAnnotation:[]};for(var a in e){t.enum.push(a);t.enumAnnotation.push({label:e[a].name})}return t}function W(e){var t=(0,i.getHomepageCollectionId)(o.default.getState());var a=h.SiteNavigationStore.getPathToCollection(t);var n=(0,s.default)(a,-2);if(!n||!n.collectionId)return false;return n.collectionId===e}function Y(e){var t=o.default.getState();var a=(0,i.getHomepageCollectionId)(t);var n={items:Object.values((0,i.getInitialSiteLayout)(t))};var r=I(n,function(e,t){if(e.collectionId===a)return t});var s=P(n,r);if(!s||!s.collectionId)return false;return s.collectionId===e}function J(e){return(0,h.isReactPagesPanelEnabled)()?Y(e):W(e)}function z(e){return h.BetaFeaturesUtils.isFeatureEnabled(c.default.NESTED_CATEGORIES)&&(0,b.default)(e)}var K=function e(t){var a=(0,g.getTemplateConfig)(t);return null===a||void 0===a?void 0:a.variation};t.isVariation=K;function Q(e){return{supportsVideoBackground:S(e),isAdjustablePageSizeCollection:A(e),isV8:(0,h.isV8)(),hasAccessPermissions:E(),canDuplicatePage:F(e),hasSystemBlogCollectionSettings:k(e),hasProductQuickView:_(e),hasOwnRelatedItemsSetting:x(e),hasEventView:L(e),canDeletePage:U(e),showPageVisibilityFields:y(),supportsInjectableCodePerItem:R(e),canEditPageLayout:N(),parentIndexType:D(e),isSevenOnePage:B(e),showCodeInjectionUpsell:M(),isSevenOne:h.TemplateVersionUtils.isSevenOne(),hasNestedCategories:z(e),isCollectionUnderMemberAreas:!!T(e.id),isMAHomepage:V(e)}}}}]);
//# sourceMappingURL=https://sourcemaps.squarespace.net/universal/scripts-compressed/3-51c73b8052e0488f3b0c7-min.en-US.js.map