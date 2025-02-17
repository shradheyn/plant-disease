from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('trained_plant_disease_model.keras')

# Ensure this path exists and is writable
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary containing disease details
disease_dic = ({
    'Apple___Apple_scab': {
        'crop': 'Apple',
        'disease': 'Apple Scab',
        'cause': """1. Apple scab overwinters primarily in fallen leaves and in the soil. Disease development is favored by wet, cool weather that generally occurs in spring and early summer.
        2. Fungal spores are carried by wind, rain or splashing water from the ground to flowers, leaves or fruit. During damp or rainy periods, newly opening apple leaves are extremely susceptible to infection. The longer the leaves remain wet, the more severe the infection will be. Apple scab spreads rapidly between 55-75 degrees Fahrenheit.""",
        'precautions': [
            'Choose resistant varieties when possible.',
            'Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.',
            'Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.',
            'Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.'
        ]
    },
    'Apple___Black_rot': {
        'crop': 'Apple',
        'disease': 'Black Rot',
        'cause': """Black rot is caused by the fungus Diplodia seriata (syn Botryosphaeria obtusa).The fungus can infect dead tissue as well as living trunks, branches, leaves and fruits. In wet weather, spores are released from these infections and spread by wind or splashing water. The fungus infects leaves and fruit through natural openings or minor wounds.""",
        'precautions': [
            'Prune out dead or diseased branches.',
            'Remove infected plant material from the area.',
            'Be sure to remove the stumps of any apple trees you cut down. Dead stumps can be a source of spores.'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'crop': 'Apple',
        'disease': 'Cedar Apple Rust',
        'cause': """Cedar apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease that depends on two species to spread and develop. It spends a portion of its two-year life cycle on Eastern red cedar (Juniperus virginiana). The pathogen’s spores develop in late fall on the juniper as a reddish brown gall on young branches of the trees.""",
        'precautions': [
            'Since the juniper galls are the source of the spores that infect the apple trees, cutting them is a sound strategy if there aren’t too many of them.',
            'While the spores can travel for miles, most of the ones that could infect your tree are within a few hundred feet.',
            'The best way to do this is to prune the branches about 4-6 inches below the galls.'
        ]
    },
    'Apple___healthy': {
        'crop': 'Apple',
        'disease': 'No disease',
        'cause': "Don't worry. Your crop is healthy. Keep it up!!!",
        'precautions': []
    },

    'Blueberry___healthy': {
        'crop': 'Blueberry',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
            'Don\'t worry. Your crop is healthy. Keep it up !!!'
        ]
    },

    'Cherry_(including_sour)___Powdery_mildew': {
        'crop': 'Cherry',
        'disease': 'Powdery Mildew',
        'cause': 'Podosphaera clandestina, a fungus that most commonly infects young, expanding leaves but can also be found on buds, fruit, and fruit stems. It overwinters as small, round, black bodies (chasmothecia) on dead leaves, on the orchard floor, or in tree crotches. Colonies produce more (asexual) spores generally around shuck fall and continue the disease cycle.',
        'precautions': [
            'Remove and destroy sucker shoots.',
            'Keep irrigation water off developing fruit and leaves by using irrigation that does not wet the leaves. Also, keep irrigation sets as short as possible.',
            'Follow cultural practices that promote good air circulation, such as pruning, and moderate shoot growth through judicious nitrogen management.'
        ]
    },

    'Cherry_(including_sour)___healthy': {
        'crop': 'Cherry',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
            'Don\'t worry. Your crop is healthy. Keep it up !!!'
        ]
    },

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'crop': 'Corn',
        'disease': 'Grey Leaf Spot',
        'cause': 'Gray leaf spot lesions on corn leaves hinder photosynthetic activity, reducing carbohydrates allocated towards grain fill. The extent to which gray leaf spot damages crop yields can be estimated based on the extent to which leaves are infected relative to grainfill. Damage can be more severe when developing lesions progress past the ear leaf around pollination time. Because a decrease in functioning leaf area limits photosynthates dedicated towards grainfill, the plant might mobilize more carbohydrates from the stalk to fill kernels.',
        'precautions': [
            'In order to best prevent and manage corn grey leaf spot, the overall approach is to reduce the rate of disease growth and expansion.',
            'This is done by limiting the amount of secondary disease cycles and protecting leaf area from damage until after corn grain formation.',
            'High risk factors for grey leaf spot in corn:',
            'a. Susceptible hybrid',
            'b. Continuous corn',
            'c. Late planting date',
            'd. Minimum tillage systems',
            'e. Field history of severe disease',
            'f. Early disease activity (before tasseling)',
            'g. Irrigation',
            'h. Favorable weather forecast for disease.'
        ]
    },

    'Corn_(maize)___Common_rust': {
        'crop': 'Corn (maize)',
        'disease': 'Common Rust',
        'cause': 'Common corn rust, caused by the fungus Puccinia sorghi, is the most frequently occurring of the two primary rust diseases of corn in the U.S., but it rarely causes significant yield losses in Ohio field (dent) corn. Occasionally field corn, particularly in the southern half of the state, does become severely affected when weather conditions favor the development and spread of rust fungus.',
        'precautions': [
            'Although rust is frequently found on corn in Ohio, very rarely has there been a need for fungicide applications. This is due to the fact that there are highly resistant field corn hybrids available and most possess some degree of resistance.',
            'However, popcorn and sweet corn can be quite susceptible. In seasons where considerable rust is present on the lower leaves prior to silking and the weather is unseasonably cool and wet, an early fungicide application may be necessary for effective disease control. Numerous fungicides are available for rust control.'
        ]
    },

    'Corn_(maize)___Northern_Leaf_Blight': {
        'crop': 'Corn (maize)',
        'disease': 'Northern Leaf Blight',
        'cause': 'Northern corn leaf blight (NCLB) is a foliar disease of corn (maize) caused by Exserohilum turcicum, the anamorph of the ascomycete Setosphaeria turcica. With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids.',
        'precautions': [
            'Management of NCLB can be achieved primarily by using hybrids with resistance, but because resistance may not be complete or may fail, it is advantageous to utilize an integrated approach with different cropping practices and fungicides.',
            'Scouting fields and monitoring local conditions is vital to control this disease.'
        ]
    },

    'Grape___Black_rot': {
    'crop': 'Grape',
    'disease': 'Black Rot',
    'cause': 'The black rot fungus overwinters in canes, tendrils, and leaves on the grape vine and on the ground. Mummified berries on the ground or those that are still clinging to the vines become the major infection source the following spring. During rain, microscopic spores (ascospores) are shot out of numerous, black fruiting bodies (perithecia) and are carried by air currents to young, expanding leaves. In the presence of moisture, these spores germinate in 36 to 48 hours and eventually penetrate the leaves and fruit stems. The infection becomes visible after 8 to 25 days. When the weather is wet, spores can be released the entire spring and summer providing continuous infection.',
    'precautions': [
        'Space vines properly and choose a planting site where the vines will be exposed to full sun and good air circulation. Keep the vines off the ground and ensure they are properly tied, limiting the amount of time the vines remain wet thus reducing infection.',
        'Keep the fruit planting and surrounding areas free of weeds and tall grass. This practice will promote lower relative humidity and rapid drying of vines and thereby limit fungal infection.',
        'Use protective fungicide sprays. Pesticides registered to protect the developing new growth include copper, captan, ferbam, mancozeb, maneb, triadimefon, and ziram. Important spraying times are as new shoots are 2 to 4 inches long, and again when they are 10 to 15 inches long, just before bloom, just after bloom, and when the fruit has set.'
    ]
    },

    'Corn_(maize)___healthy': {
    'crop': 'Corn (maize)',
    'disease': 'No disease',
    'cause': '',
    'precautions': [
        'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Grape___Esca(Black_Measles)': {
    'crop': 'Grape',
    'disease': 'Black Measles',
    'cause': 'Black Measles is caused by a complex of fungi that includes several species of Phaeoacremonium, primarily by P. aleophilum (currently known by the name of its sexual stage, Togninia minima), and by Phaeomoniella chlamydospora. The overwintering structures that produce spores (perithecia or pycnidia, depending on the pathogen) are embedded in diseased woody parts of vines. During fall to spring rainfall, spores are released and wounds made by dormant pruning provide infection sites. Wounds may remain susceptible to infection for several weeks after pruning with susceptibility declining over time.',
    'precautions': [
        'Post-infection practices (sanitation and vine surgery) for use in diseased, mature vineyards are not as effective and are far more costly than adopting preventative practices (delayed pruning, double pruning, and applications of pruning-wound protectants) in young vineyards.',
        'Sanitation and vine surgery may help maintain yields. In spring, look for dead spurs or for stunted shoots. Later in summer, when there is a reduced chance of rainfall, practice good sanitation by cutting off these cankered portions of the vine beyond the canker, to where wood appears healthy. Then remove diseased, woody debris from the vineyard and destroy it.',
        'The fungicides labeled as pruning-wound protectants, consider using alternative materials, such as a wound sealant with 5 percent boric acid in acrylic paint (Tech-Gro B-Lock), which is effective against Eutypa dieback and Esca, or an essential oil (Safecoat VitiSeal).'
    ]
    },

    'Grape___Leaf_blight(Isariopsis_Leaf_Spot)': {
    'crop': 'Grape',
    'disease': 'Leaf Blight',
    'cause': 'Apple scab overwinters primarily in fallen leaves and in the soil. Disease development is favored by wet, cool weather that generally occurs in spring and early summer. Fungal spores are carried by wind, rain or splashing water from the ground to flowers, leaves or fruit. During damp or rainy periods, newly opening apple leaves are extremely susceptible to infection. The longer the leaves remain wet, the more severe the infection will be. Apple scab spreads rapidly between 55-75 degrees Fahrenheit.',
    'precautions': [
        'Choose resistant varieties when possible.',
        'Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.',
        'Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.',
        'Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.'
    ]
    },

    'Orange___Haunglongbing(Citrus_greening)': {
    'crop': 'Orange',
    'disease': 'Citrus Greening',
    'cause': 'Huanglongbing (HLB) or citrus greening is the most severe citrus disease, currently devastating the citrus industry worldwide. The presumed causal bacterial agent Candidatus Liberibacter spp. affects tree health as well as fruit development, ripening, and quality of citrus fruits and juice.',
    'precautions': [
        'In regions where disease incidence is low, the most common practices are avoiding the spread of infection by removal of symptomatic trees, protecting grove edges through intensive monitoring, use of pesticides, and biological control of the vector ACP.',
        'According to Singerman and Useche (2016), CHMAs coordinate insecticide application to control the ACP spreading across area-wide neighboring commercial citrus groves as part of a plan to address the HLB disease.',
        'In addition to foliar nutritional sprays, plant growth regulators were tested, unsuccessfully, to reduce HLB-associated fruit drop (Albrigo and Stover, 2015).'
    ]
    },

    'Peach___Bacterial_spot': {
        'crop': 'Peach',
        'disease': 'Bacterial Spot',
        'cause': 'The disease is caused by four species of Xanthomonas (X. euvesicatoria, X. gardneri, X. perforans, and X. vesicatoria). In North Carolina, X. perforans is the predominant species associated with bacterial spot on tomato and X. euvesicatoria is the predominant species associated with the disease on pepper. All four bacteria are strictly aerobic, gram-negative rods with a long whip-like flagellum (tail) that allows them to move in water, which allows them to invade wet plant tissue and cause infection.',
        'precautions': [
    'The most effective management strategy is the use of pathogen-free certified seeds and disease-free transplants to prevent the introduction of the pathogen into greenhouses and field production areas. Inspect plants very carefully and reject infected transplants- including your own!',
    'In transplant production greenhouses, minimize overwatering and handling of seedlings when they are wet.',
    'Trays, benches, tools, and greenhouse structures should be washed and sanitized between seedlings crops.',
    'Do not spray, tie, harvest, or handle wet plants as that can spread the disease.'
    ]
    },

    'Pepper,bell__Bacterial_spot': {
        'crop': 'Pepper',
        'disease': 'Bacterial Spot',
        'cause': 'Bacterial spot is caused by several species of gram-negative bacteria in the genus Xanthomonas. In culture, these bacteria produce yellow, mucoid colonies. A "mass" of bacteria can be observed oozing from a lesion by making a cross-sectional cut through a leaf lesion, placing the tissue in a droplet of water, placing a cover-slip over the sample, and examining it with a microscope (~200X).',
        'precautions': [
    'The primary management strategy of bacterial spot begins with use of certified pathogen-free seed and disease-free transplants.',
    'The bacteria do not survive well once host material has decayed, so crop rotation is recommended. Once the bacteria are introduced into a field or greenhouse, the disease is very difficult to control.',
    'Pepper plants are routinely sprayed with copper-containing bactericides to maintain a "protective" cover on the foliage and fruit.'
    ]
    },

    'Peach___healthy': {
        'crop': 'Peach',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Pepper,bell__healthy': {
        'crop': 'Pepper',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Potato___healthy': {
        'crop': 'Potato',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Raspberry___healthy': {
        'crop': 'Raspberry',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Soybean___healthy': {
        'crop': 'Soybean',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Strawberry___healthy': {
        'crop': 'Strawberry',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Tomato___healthy': {
        'crop': 'Tomato',
        'disease': 'No disease',
        'cause': '',
        'precautions': [
    'Don\'t worry. Your crop is healthy. Keep it up !!!'
    ]
    },

    'Potato___Early_blight': {
        'crop': 'Potato',
        'disease': 'Early Blight',
        'cause': 'Early blight (EB) is a disease of potato caused by the fungus Alternaria solani. It is found wherever potatoes are grown. The disease primarily affects leaves and stems, but under favorable weather conditions, and if left uncontrolled, can result in considerable defoliation and enhance the chance for tuber infection. Premature defoliation may lead to considerable reduction in yield. Primary infection is difficult to predict since EB is less dependent upon specific weather conditions than late blight.',
        'precautions': [
    'Plant only disease-free, certified seed.',
    'Follow a complete and regular foliar fungicide spray program.',
    'Practice good killing techniques to lessen tuber infections.',
    'Allow tubers to mature before digging, dig when vines are dry, not wet, and avoid excessive wounding of potatoes during harvesting and handling.'
    ]
    },

    'Potato___Late_blight': {
        'crop': 'Potato',
        'disease': 'Late Blight',
        'cause': 'Late blight is a potentially devastating disease of potato, infecting leaves, stems and fruits of plants. The disease spreads quickly in fields and can result in total crop failure if untreated. Late blight of potato was responsible for the Irish potato famine of the late 1840s. Late blight is caused by the oomycete Phytophthora infestans. Oomycetes are fungus-like organisms also called water molds, but they are not true fungi. There are many different strains of P. infestans, called clonal lineages and designated by a number code (i.e. US-23). Many clonal lineages affect both tomato and potato, but some lineages are specific to one host or the other. The host range is typically limited to potato and tomato, but hairy nightshade (Solanum physalifolium) is a closely related weed that can readily become infected and may contribute to disease spread. Under ideal conditions, such as a greenhouse, petunia also may become infected.',
        'precautions': [
    'Seed infection is unlikely on commercially prepared tomato seed or on saved seed that has been thoroughly dried.',
    'Inspect tomato transplants for late blight symptoms prior to purchase and/or planting, as tomato transplants shipped from southern regions may be infected.',
    'If infection is found in only a few plants within a field, infected plants should be removed, disced-under, killed with herbicide or flame-killed to avoid spreading through the entire field.'
    ]
    },

    'Squash___Powdery_mildew': {
        'crop': 'Squash',
        'disease': 'Powdery Mildew',
        'cause': 'Powdery mildew infections favor humid conditions with temperatures around 68-81° F. In warm, dry conditions, new spores form and easily spread the disease. Symptoms of powdery mildew first appear mid to late summer. The older leaves are more susceptible and powdery mildew will infect them first. Wind blows spores produced in leaf spots to infect other leaves. Under favorable conditions, powdery mildew can spread very rapidly, often covering all of the leaves.',
        'precautions': [
    'Apply fertilizer based on soil test results. Avoid over-applying nitrogen.',
    'Provide good air movement around plants through proper spacing, staking of plants and weed control.',
    'Once a week, examine five mature leaves for powdery mildew infection. In large plantings, repeat at 10 different locations in the field.',
    'If susceptible varieties are growing in an area where powdery mildew has resulted in yield loss in the past, fungicide may be necessary.'
    ]
    },

    'Strawberry___Leaf_scorch': {
        'crop': 'Strawberry',
        'disease': 'Leaf Scorch',
        'cause': 'Scorched strawberry leaves are caused by a fungal infection which affects the foliage of strawberry plantings. The fungus responsible is called Diplocarpon earliana. Strawberries with leaf scorch may first show signs of issue with the development of small purplish blemishes that occur on the topside of leaves.',
        'precautions': [
    'Since this fungal pathogen overwinters on the fallen leaves of infected plants, proper garden sanitation is key.',
    'This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants.',
    'The avoidance of waterlogged soil and frequent garden cleanup will help to reduce the likelihood of spread of this fungus.'
    ]
    },

    'Tomato___Bacterial_spot': {
        'crop': 'Tomato',
        'disease': 'Bacterial Spot',
        'cause': 'The disease is caused by four species of Xanthomonas (X. euvesicatoria, X. gardneri, X. perforans, and X. vesicatoria). In North Carolina, X. perforans is the predominant species associated with bacterial spot on tomato and X. euvesicatoria is the predominant species associated with the disease on pepper. All four bacteria are strictly aerobic, gram-negative rods with a long whip-like flagellum (tail) that allows them to move in water, which allows them to invade wet plant tissue and cause infection.',
        'precautions': [
    'The most effective management strategy is the use of pathogen-free certified seeds and disease-free transplants to prevent the introduction of the pathogen into greenhouses and field production areas. Inspect plants very carefully and reject infected transplants- including your own!',
    'In transplant production greenhouses, minimize overwatering and handling of seedlings when they are wet.',
    'Trays, benches, tools, and greenhouse structures should be washed and sanitized between seedlings crops.',
    'Do not spray, tie, harvest, or handle wet plants as that can spread the disease.'
    ]
    },

    'Tomato___Early_blight': {
        'crop': 'Tomato',
        'disease': 'Early Blight',
        'cause': 'Early blight can be caused by two different closely related fungi, Alternaria tomatophila and Alternaria solani. Alternaria tomatophila is more virulent on tomato than A. solani, so in regions where A. tomatophila is found, it is the primary cause of early blight on tomato. However, if A. tomatophila is absent, A. solani will cause early blight on tomato.',
        'precautions': [
    'Use pathogen-free seed, or collect seed only from disease-free plants.',
    'Rotate out of tomatoes and related crops for at least two years.',
    'Control susceptible weeds such as black nightshade and hairy nightshade, and volunteer tomato plants throughout the rotation.',
    'Fertilize properly to maintain vigorous plant growth. Particularly, do not over-fertilize with potassium and maintain adequate levels of both nitrogen and phosphorus.',
    'Avoid working in plants when they are wet from rain, irrigation, or dew.',
    'Use drip irrigation instead of overhead irrigation to keep foliage dry.'
    ]
    },

    'Tomato___Late_blight': {
        'crop': 'Tomato',
        'disease': 'Late Blight',
        'cause': 'Late blight is a potentially devastating disease of tomato, infecting leaves, stems, and fruits of plants. The disease spreads quickly in fields and can result in total crop failure if untreated. Late blight is caused by the oomycete Phytophthora infestans. Oomycetes are fungus-like organisms also called water molds, but they are not true fungi. There are many different strains of P. infestans, called clonal lineages and designated by a number code (i.e. US-23). Many clonal lineages affect both tomato and potato, but some lineages are specific to one host or the other. The host range is typically limited to potato and tomato, but hairy nightshade (Solanum physalifolium) is a closely related weed that can readily become infected and may contribute to disease spread. Under ideal conditions, such as a greenhouse, petunia also may become infected.',
        'precautions': [
    'Inspect tomato transplants for late blight symptoms prior to purchase and/or planting, as tomato transplants shipped from southern regions may be infected.',
    'If infection is found in only a few plants within a field, infected plants should be removed, disced-under, killed with herbicide or flame-killed to avoid spreading through the entire field.'
    ]
    },

    'Tomato___Leaf_Mold': {
        'crop': 'Tomato',
        'disease': 'Leaf Mold',
        'cause': 'Leaf mold is caused by the fungus Passalora fulva (previously called Fulvia fulva or Cladosporium fulvum). It is not known to be pathogenic on any plant other than tomato. Leaf spots grow together and turn brown. Leaves wither and die but often remain attached to the plant. Fruit infections start as a smooth black irregular area on the stem end of the fruit. As the disease progresses, the infected area becomes sunken, dry, and leathery.',
        'precautions': [
    'Use drip irrigation and avoid watering foliage.',
    'Space plants to provide good air movement between rows and individual plants.',
    'Stake, string, or prune to increase airflow in and around the plant.',
    'Sterilize stakes, ties, trellises, etc. with 10 percent household bleach or commercial sanitizer.',
    'Circulate air in greenhouses or tunnels with vents and fans and by rolling up high tunnel sides to reduce humidity around plants.',
    'Keep night temperatures in greenhouses higher than outside temperatures to avoid dew formation on the foliage.',
    'Remove crop residue at the end of the season. Burn it or bury it away from tomato production areas.'
    ]
    },

    'Tomato___Septoria_leaf_spot': {
        'crop': 'Tomato',
        'disease': 'Leaf Spot',
        'cause': 'Septoria leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods.',
        'precautions': [
    'Remove diseased leaves.',
    'Improve air circulation around the plants.',
    'Mulch around the base of the plants.',
    'Do not use overhead watering.',
    'Use fungicidal sprays.'
    ]
    },

    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'crop': 'Tomato',
        'disease': 'Two-spotted Spider Mite',
        'cause': 'The two-spotted spider mite is the most common mite species that attacks vegetable and fruit crops. They have up to 20 generations per year and are favored by excess nitrogen and dry and dusty conditions. Outbreaks are often caused by the use of broad-spectrum insecticides which interfere with the numerous natural enemies that help to manage mite populations.',
        'precautions': [
    'Avoid early season, broad-spectrum insecticide applications for other pests.',
    'Do not over-fertilize.',
    'Overhead irrigation or prolonged periods of rain can help reduce populations.'
    ]
    },

    'Tomato___Target_Spot': {
        'crop': 'Tomato',
        'disease': 'Target Spot',
        'cause': 'The fungus causes plants to lose their leaves; it is a major disease. If infection occurs before the fruit has developed, yields are low. This is a common disease on tomato in Pacific island countries. The disease occurs in the screen house and in the field.',
        'precautions': [
    'Remove a few branches from the lower part of the plants to allow better airflow at the base.',
    'Remove and burn the lower leaves as soon as the disease is seen, especially after the lower fruit trusses have been picked.',
    'Keep plots free from weeds, as some may be hosts of the fungus.',
    'Do not use overhead irrigation; otherwise, it will create conditions for spore production and infection.'
    ]
    },

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'crop': 'Tomato',
        'disease': 'Yellow Leaf Curl Virus',
        'cause': 'TYLCV is transmitted by the insect vector Bemisia tabaci in a persistent-circulative nonpropagative manner. The virus can be efficiently transmitted during the adult stages. This virus transmission has a short acquisition access period of 15–20 minutes, and latent period of 8–24 hours.',
        'precautions': [
    'Currently, the most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties.',
    'The effectiveness of insecticides is not optimal in tropical areas due to whitefly resistance against the insecticides; therefore, insecticides should be alternated or mixed to provide the most effective treatment against virus transmission.',
    'Other methods to control the spread of TYLCV include planting resistant/tolerant lines, crop rotation, and breeding for resistance of TYLCV. As with many other plant viruses, one of the most promising methods to control TYLCV is the production of transgenic tomato plants resistant to TYLCV.'
    ]
    },

    'Tomato___Tomato_mosaic_virus': {
        'crop': 'Tomato',
        'disease': 'Mosaic Virus',
        'cause': 'Tomato mosaic virus and tobacco mosaic virus can exist for two years in dry soil or leaf debris, but will only persist one month if soil is moist. The viruses can also survive in infected root debris in the soil for up to two years. Seed can be infected and pass the virus to the plant but the disease is usually introduced and spread primarily through human activity. The virus can easily spread between plants on workers\' hands, tools, and clothes with normal activities such as plant tying, removing of suckers, and harvest. The virus can even survive the tobacco curing process, and can spread from cigarettes and other tobacco products to plant material handled by workers after a cigarette.',
        'precautions': [
    'Purchase transplants only from reputable sources. Ask about the sanitation procedures they use to prevent disease.',
    'Inspect transplants prior to purchase. Choose only transplants showing no clear symptoms.',
    'Avoid planting in fields where tomato root debris is present, as the virus can survive long-term in roots.',
    'Wash hands with soap and water before and during the handling of plants to reduce potential spread between plants.'
    ]
    }
    })

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    result_index = model_prediction(file_path)
    os.remove(file_path)  # Clean up after prediction

    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    prediction = class_names[result_index]
    disease_details = disease_dic.get(prediction, {})
    
    return render_template('predict.html', prediction=prediction, details=disease_details)

def model_prediction(test_image_path):
    image = load_img(test_image_path, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

if __name__ == '__main__':
    app.run(debug=True)
